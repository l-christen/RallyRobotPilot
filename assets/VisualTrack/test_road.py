import bpy
import bmesh
from mathutils import Vector
from math import atan2, cos, sin, sqrt, pi


ROAD_HALF_WIDTH = 1
TURN_RADIUS = 1
# 
TURN_SUBDIVISION = 5

def main():
    print("#######################################")
    print("#######################################")
    print("#######################################")
    selected = bpy.context.selected_objects
    
    if len(selected) != 1:
        print("Must have selected only one object")
        return
    
    #   Get tempalte mesh
    template_road = selected[0]
    template_mesh = template_road.data
    
    
    vertex_data = [None] * len(template_mesh.vertices)
    edge_data = {}
    for edge in template_mesh.edges:
        edge_data[(edge.vertices[1], edge.vertices[0])] = {}
        edge_data[(edge.vertices[0], edge.vertices[1])] = {}
    
    for vertex in template_mesh.vertices:
        connected_edges = []
                
        #   Collect inbound edges
        for edge in template_mesh.edges:            
            if edge.vertices[0] == vertex.index:
                connected_edges.append(((edge.vertices[0], edge.vertices[1]), edge))
            elif edge.vertices[1] == vertex.index:
                connected_edges.append(((edge.vertices[1], edge.vertices[0]), edge))
                
        #   Skiè disconnected vertices
        if len(connected_edges) == 0:
            continue
                
        # Do not process end points yet
        if len(connected_edges) == 1:
            (vid1,vid2),_ = connected_edges[0]
            
            dir = template_mesh.vertices[vid2].co - template_mesh.vertices[vid1].co
            a = atan2(dir.y, dir.x)
            
            edge_data[(vid1, vid2)]["pt_above"] = Vector((cos(a+pi/2), sin(a+pi/2), 0)) * ROAD_HALF_WIDTH + template_mesh.vertices[vid1].co
            edge_data[(vid1, vid2)]["pt_below"] = Vector((cos(a-pi/2), sin(a-pi/2), 0)) * ROAD_HALF_WIDTH + template_mesh.vertices[vid1].co
            continue
        
        #   Inner turn but no intersections
        #if len(connected_edges) == 2:
        #    continue        
        
        angles = [0] * len(connected_edges)
        side_lengths = [0] * len(connected_edges)
        for eid, ((vid1, vid2),edge) in enumerate(connected_edges):
            
            dir = template_mesh.vertices[vid2].co - template_mesh.vertices[vid1].co
            angles[eid] = atan2(dir.y, dir.x)
            side_lengths[eid] = sqrt(dir.x*dir.x+dir.y*dir.y)
            if angles[eid] < 0:
                angles[eid] += 2*pi
        
        
        # Sort edge based on outbound angle so we know neighboring edges
        sorted_pairs = [(a,e) for a,e in sorted(zip(angles,connected_edges))]
        sorted_angles, sorted_edges = zip(*sorted_pairs)
        
        
        junctions_pts = []
        for (a1,e1),(a2,e2) in zip(sorted_pairs, sorted_pairs[1:] + sorted_pairs[:1]): 
            #   Handle wrap arond numerically
            if a2 < a1:
                a2 += 2*pi
            
            diff_angle = a2 - a1
            center_out_turn = diff_angle < pi
            dir  = Vector((cos(diff_angle), sin(diff_angle), 0))
            # Negative sign to orient the turn inward
            norm = Vector((cos(diff_angle - 3.14159/2), sin(diff_angle - 3.14159/2), 0))
            
            #   Don't do computation for inner center for intermediate case
            if len(connected_edges) == 2 and not center_out_turn:
                continue
            
            # Compute the intersection point of 2 lines (corresponding to road centers) with the road+turn radius offset
            offset = 0
            if abs(dir.y) < 1e-6:
                distAlongX = 0
                offset = ROAD_HALF_WIDTH
            else:
                offset = ROAD_HALF_WIDTH + TURN_RADIUS if center_out_turn else ROAD_HALF_WIDTH - TURN_RADIUS
                distAlong2 = -(norm.y * (offset) - (offset)) / dir.y
                distAlongX = distAlong2 * dir.x + norm.x * offset
            
            center_dist = sqrt(distAlongX**2 + offset**2)
            center_angle = atan2(offset, distAlongX)
            
            #   Position of the center circle relative to the vertex
            center = Vector((cos(a1 + center_angle) * center_dist, sin(a1 + center_angle) * center_dist, 0)) + vertex.co
            
            #   Sample points along circle to make junction
            start_angle = 2*pi - pi/2 if center_out_turn else pi/2
            end_angle = diff_angle + pi/2 if center_out_turn else diff_angle - pi/2
            angle_range = end_angle - start_angle
            
            
            turn_center = vertex.co
            
            if len(connected_edges) > 2:
                junction_pts = []
                for i in range(TURN_SUBDIVISION+1):
                    ratio = i / TURN_SUBDIVISION
                    on_circle = Vector((cos(angle_range * ratio + start_angle + a1), sin(angle_range * ratio + start_angle + a1), 0)) * TURN_RADIUS
                    junction_pts.append(on_circle + center)
                
                edge_data[(e1[0][0], e1[0][1])]["pt_above"] = junction_pts[0]
                edge_data[(e2[0][0], e2[0][1])]["pt_below"] = junction_pts[-1]
                
                junctions_pts.append(junction_pts)
                turn_center = vertex.co
            elif len(connected_edges) == 2:
                
                in_junction_pts = []
                out_junction_pts = []
                for i in range(TURN_SUBDIVISION+1):
                    ratio = i / TURN_SUBDIVISION
                    in_circle = Vector((cos(angle_range * ratio + start_angle + a1), sin(angle_range * ratio + start_angle + a1), 0)) * TURN_RADIUS
                    in_junction_pts.append(in_circle + center)
                    
                    out_circle = Vector((cos(angle_range * ratio + start_angle + a1), sin(angle_range * ratio + start_angle + a1), 0)) * (TURN_RADIUS + ROAD_HALF_WIDTH * 2)
                    out_junction_pts.append(out_circle + center)
                
                turn_center = Vector((cos(angle_range * 0.5 + start_angle + a1), sin(angle_range * 0.5 + start_angle + a1), 0)) * (TURN_RADIUS + ROAD_HALF_WIDTH) + center
                
                edge_data[(e1[0][0], e1[0][1])]["pt_above"] = in_junction_pts[0]
                edge_data[(e1[0][0], e1[0][1])]["pt_below"] = out_junction_pts[0]
                edge_data[(e2[0][0], e2[0][1])]["pt_below"] = in_junction_pts[-1]
                edge_data[(e2[0][0], e2[0][1])]["pt_above"] = out_junction_pts[-1]
                
                junctions_pts.append(in_junction_pts)
                junctions_pts.append(out_junction_pts[::-1])
                
            
                
        vertex_data[vertex.index] = {"junctions_pts":junctions_pts, "center":turn_center}
        
    road_vertices = []
    road_faces = []
    
    collision_vertices = []
    collision_edges = []
    
    print(edge_data)
    #   Mesh road segments
    for edge in template_mesh.edges:
        vid1,vid2 = edge.vertices
        
        v1 = edge_data[(vid1, vid2)]["pt_above"]
        v2 = edge_data[(vid1, vid2)]["pt_below"] # Edge oreintation inverse below and above
        v3 = edge_data[(vid2, vid1)]["pt_above"]
        v4 = edge_data[(vid2, vid1)]["pt_below"]
        
        idx = len(road_vertices)
        road_vertices += [v1,v2,v3,v4]
        road_faces += [(idx, idx +1, idx+2, idx+3)]
        
        cid = len(collision_vertices)
        collision_vertices += [v1,v2,v3,v4]
        collision_edges += [(cid,cid+3),(cid+1,cid+2)]
        
    #   Mesh intersections
    for vertex in template_mesh.vertices:
        vid = vertex.index
        
        if vertex_data[vid] is None or "junctions_pts" not in vertex_data[vid]:
            continue
        
        junction_data = vertex_data[vid]["junctions_pts"]
        
        ovid = len(road_vertices)
        road_vertices.append(vertex_data[vid]["center"])
        nbr_pts = 0        
        
        for junction in junction_data:
            nbr_pts += len(junction)
            road_vertices += junction
            
            #   Do it for collision detection
            cid = len(collision_vertices)
            print(cid)
            collision_vertices += junction
            id1 = [e+cid for e in range(len(junction)-1)]
            id2 = [e+1 for e in id1]
            print(list(zip(id1, id2)))
            collision_edges += zip(id1, id2)
        
                
        #   Connect the points
        for i in range(nbr_pts):
            road_faces += [(ovid, ovid+1 + i, ovid +1 + (i + 1) % nbr_pts)]
            
    
    def cleanObjectMesh(name):
        if name in bpy.data.meshes:
            bpy.data.meshes.remove(bpy.data.meshes[name])
            
    cleanObjectMesh("Road")
    road_mesh = bpy.data.meshes.new('Road')
    road_obj = bpy.data.objects.new(road_mesh.name, road_mesh)
    coll = bpy.data.collections["Procedural"]
    coll.objects.link(road_obj)
    road_obj.parent = template_road
    road_mesh.from_pydata(road_vertices, [], road_faces)
            
    cleanObjectMesh("RoadWalls")
    collision_mesh = bpy.data.meshes.new('RoadWalls')
    collision_obj = bpy.data.objects.new(collision_mesh.name, collision_mesh)
    coll = bpy.data.collections["Procedural"]
    coll.objects.link(collision_obj)
    collision_obj.parent = template_road
    collision_mesh.from_pydata(collision_vertices, collision_edges, [])
    collision_obj.modifiers.new("GeomNodes", 'NODES')
    geomnodes = bpy.data.node_groups["ProceduralRoadWall"]
    collision_obj.modifiers["GeomNodes"].node_group = geomnodes
    

if __name__ == "__main__":
    main()

