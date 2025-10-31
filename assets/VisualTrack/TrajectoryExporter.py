
#
# Exportation tool to extract trajectories and write them to a json file
#   Trajectories are loops in a mesh file made of a closed path between vertices and edges.
#     On trajectory mesh can contain multiple trajectories but trajectories cannot share edges or vertices
#

import bpy
import bmesh
from mathutils import Vector
from math import atan2, cos, sin, sqrt, pi
import json


ROAD_HALF_WIDTH = 1
TURN_RADIUS = 1
# 
TURN_SUBDIVISION = 5

def write_trajectories(file_path):
    print("#######################################")
    print("#######################################")
    print("#######################################")
    selected = bpy.context.selected_objects
    
    if len(selected) != 1:
        print("Must have selected only one object")
        return
    
    #   Get tempalte mesh
    template_road = selected[0]
    traj_mesh = template_road.data
    
    visited_vertices = []
    trajectories = []
    for vertex in traj_mesh.vertices:
        if vertex in visited_vertices:
            continue
        
        loop = [vertex]
        visited_edges = []
        sub_visited_vertices = []
        loop_found = False
        while not loop_found:
            no_next_found = True
            current_vertex = loop[-1]
            for e in traj_mesh.edges:
                if e in visited_edges:
                    continue
                
                next_v = 0
                if e.vertices[0] == current_vertex.index:
                    next_v = traj_mesh.vertices[e.vertices[1]]
                elif e.vertices[1] == current_vertex.index:
                    next_v = traj_mesh.vertices[e.vertices[0]]
                else:
                    continue
                
                if next_v.index == loop[0].index:
                    loop_found = True
                    break
                
                #   If next is not previ
                if len(loop) <= 1 or next_v != loop[-2]:
                    visited_edges.append(e)
                    loop.append(traj_mesh.vertices[next_v.index])
                    no_next_found = False
                    break
            
            if no_next_found or loop_found:
                break
            
        if loop_found:
            print([e.index for e in loop])
            visited_vertices += loop  
            trajectories.append(loop)
    
    pts = []
    for traj in trajectories:
        # /!\ INVERSE Y AND Z to be ursinae compatible
        pts.append(list([(v.co[0], v.co[2], v.co[1]) for v in traj]))
    
    if file_path is not None:
        data = json.dumps(list(pts))
        f = open(file_path, 'w', encoding='utf-8')
        f.write(data)
        f.close()
    return {'FINISHED'}

# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator

class TrajectoryExporter(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_test.some_data"
    bl_label = "Export trajectories"

    # ExportHelper mixin class uses this
    filename_ext = ".traj"
    filter_glob: bpy.props.StringProperty(default="*.traj", options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        return write_trajectories(self.filepath)


def menu_func_export(self, context):
    self.layout.operator(TrajectoryExporter.bl_idname, text="Trajectory exporter (.traj)")

def register():
    bpy.utils.register_class(TrajectoryExporter)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(TrajectoryExporter)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
