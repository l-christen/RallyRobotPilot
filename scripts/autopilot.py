import sys
from model.infer import create_inference_engine
from PyQt6 import QtWidgets
from data_collector import DataCollectionUI


class AutopilotProcessor:
    """
    Processeur d'autopilot pour contrôler la voiture via le modèle CNN-LSTM.
    Gère le buffering, l'inférence et l'envoi des commandes.
    """
    
    def __init__(self, checkpoint_path='checkpoints/test_best_model.pth', device='auto'):
        """
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            device: 'auto', 'cuda' ou 'cpu'
        """
        # Détection automatique du device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        # Créer le moteur d'inférence
        self.engine = create_inference_engine(
            checkpoint_path=checkpoint_path,
            seq_len=15,
            device=device
        )
        
        # État des commandes (pour ne transmettre que les changements)
        self.prev_controls = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }
        
        # Configuration
        self.threshold = 0.5  # Seuil de décision
        self.debug_interval = 50  # Afficher debug toutes les N frames
        
        print(f"[✓] Autopilot initialisé sur {self.device}")
        print(f"[✓] Seuil de décision: {self.threshold}")
    
    def process_message(self, message, data_collector):
        """
        Traite un message de sensing reçu du jeu.
        
        Args:
            message: SensingSnapshot contenant l'image et les données
            data_collector: Interface pour envoyer les commandes
        """
        try:
            # Ajouter la frame au buffer
            is_ready = self.engine.add_frame(message.image)
            
            # Afficher la progression pendant le warm-up
            if not is_ready:
                status = self.engine.get_status()
                if status['buffer_size'] % 5 == 0 or status['buffer_size'] == 1:
                    print(f"[WARM-UP] {status['buffer_size']}/{status['required_size']} frames "
                          f"({status['progress']*100:.0f}%)")
                return
            
            # Effectuer l'inférence
            result = self.engine.predict(threshold=self.threshold)
            
            if result is None:
                # Ne devrait pas arriver mais sécurité
                return
            
            # Extraire les contrôles
            controls_list = result['controls']  # [fwd, back, left, right]
            probabilities = result['probabilities']
            
            commands = {
                "forward": controls_list[0],
                "back": controls_list[1],
                "left": controls_list[2],
                "right": controls_list[3],
            }
            
            # Debug périodique
            if self.engine.total_inferences % self.debug_interval == 0:
                self._log_debug(probabilities, commands, result)
            
            # Envoyer les transitions d'état au jeu
            self._send_transitions(commands, data_collector)
            
        except Exception as e:
            print(f"[X] Erreur dans l'autopilot: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_transitions(self, commands, data_collector):
        """
        Envoie uniquement les changements d'état des commandes.
        
        Args:
            commands: Dict {cmd_name: bool}
            data_collector: Interface DataCollectionUI
        """
        for cmd_name, new_state in commands.items():
            old_state = self.prev_controls[cmd_name]
            
            if new_state != old_state:
                # Transition détectée
                data_collector.onCarControlled(cmd_name, new_state)
                self.prev_controls[cmd_name] = new_state
                
                # Log la transition
                state_str = "ON " if new_state else "OFF"
                print(f"[CMD] {cmd_name:8s} → {state_str}")
    
    def _log_debug(self, probabilities, commands, result):
        """Affiche des informations de debug détaillées."""
        print("\n" + "="*60)
        print(f"[DEBUG] Inférence #{self.engine.total_inferences}")
        print("-"*60)
        print(f"Probabilités:")
        print(f"  Forward : {probabilities[0]:.3f} → {commands['forward']}")
        print(f"  Back    : {probabilities[1]:.3f} → {commands['back']}")
        print(f"  Left    : {probabilities[2]:.3f} → {commands['left']}")
        print(f"  Right   : {probabilities[3]:.3f} → {commands['right']}")
        print(f"Prédictions auxiliaires:")
        print(f"  Vitesse : {result['speed']:.2f}")
        print(f"  Raycasts: min={result['raycasts'].min():.2f}, "
              f"max={result['raycasts'].max():.2f}, "
              f"mean={result['raycasts'].mean():.2f}")
        print("="*60 + "\n")
    
    def reset(self):
        """Réinitialise l'autopilot (utile lors d'un reset de partie)."""
        self.engine.reset()
        self.prev_controls = {
            "forward": False,
            "back": False,
            "left": False,
            "right": False
        }
        print("[!] Autopilot réinitialisé")
    
    def set_threshold(self, threshold):
        """Change le seuil de décision."""
        self.threshold = max(0.0, min(1.0, threshold))
        print(f"[!] Seuil mis à jour: {self.threshold}")
    
    def get_statistics(self):
        """Retourne les statistiques d'utilisation."""
        status = self.engine.get_status()
        return {
            'total_frames_processed': status['total_frames'],
            'total_inferences': status['total_inferences'],
            'buffer_ready': status['ready'],
            'current_controls': self.prev_controls.copy()
        }


def main():
    """Point d'entrée principal."""
    
    # Hook pour afficher les exceptions PyQt
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook
    
    # Créer l'application Qt
    app = QtWidgets.QApplication(sys.argv)
    
    # Initialiser l'autopilot
    try:
        autopilot = AutopilotProcessor(
            checkpoint_path='checkpoints/test_best_model.pth',
            device='auto'
        )
    except Exception as e:
        print(f"[X] Impossible d'initialiser l'autopilot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Créer l'interface de collecte avec callback autopilot
    window = DataCollectionUI(
        message_processing_callback=autopilot.process_message
    )
    
    print("\n" + "="*60)
    print("AUTOPILOT ACTIF")
    print("="*60)
    print("Le modèle va prendre le contrôle de la voiture.")
    print("Attendez le warm-up (15 frames) avant le début du contrôle.")
    print("="*60 + "\n")
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()