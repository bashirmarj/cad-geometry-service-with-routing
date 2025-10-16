"""
CAD Geometry Analysis Microservice
Parses STEP/IGES files using pythonOCC to extract real geometry data
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import logging

# Import industrial routing modules
from routing_selector_industrial import select_routings_industrial
from machining_estimator import estimate_machining_time_and_cost

# OCC imports for STEP parsing
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core import topods
import math

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'cad-geometry-analyzer'}), 200

@app.route('/analyze-cad', methods=['POST'])
def analyze_cad():
    """
    Analyze STEP/IGES file geometry with industrial routing selection
    
    Accepts: multipart/form-data with:
        - file: CAD file (STEP/IGES)
        - material (optional): Material name (default: "Cold Rolled Steel")
        - tolerance (optional): Tolerance in mm (default: 0.02)
        - quality (optional): Mesh quality 0-1 (default: 0.999)
    
    Returns: JSON with geometry properties, detected features, routing recommendations, and cost estimates
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
    
    # Get material and tolerance from form data
    material = request.form.get('material', 'Cold Rolled Steel')
    tolerance = float(request.form.get('tolerance', 0.02))
    
    logger.info(f"Analyzing file: {file.filename}, material: {material}, tolerance: {tolerance}mm")
    
    # Save temporarily
    file_ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Parse STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_path)
        
        if status != 1:  # IFSelect_RetDone
            raise Exception(f"Failed to read STEP file, status: {status}")
        
        reader.TransferRoots()
        shape = reader.OneShape()
        
        if shape.IsNull():
            raise Exception("Failed to extract shape from STEP file")
        
        # Calculate volume
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume_mm3 = props.Mass()
        
        # Calculate surface area
        brepgprop.SurfaceProperties(shape, props)
        surface_area_mm2 = props.Mass()
        
        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        width_mm = xmax - xmin
        height_mm = ymax - ymin
        depth_mm = zmax - zmin
        
        # Analyze faces for feature detection
        cylindrical_faces = 0
        planar_faces = 0
        total_faces = 0
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            surface = BRepAdaptor_Surface(face)
            surface_type = surface.GetType()
            
            if surface_type == GeomAbs_Cylinder:
                cylindrical_faces += 1
            elif surface_type == GeomAbs_Plane:
                planar_faces += 1
            
            total_faces += 1
            explorer.Next()
        
        logger.info(f"Geometry analysis: {total_faces} faces ({cylindrical_faces} cylindrical, {planar_faces} planar)")
        
        # Determine if cylindrical (>40% cylindrical faces)
        is_cylindrical = cylindrical_faces > (total_faces * 0.4) if total_faces > 0 else False
        has_flat_surfaces = planar_faces > 2
        
        # Complexity scoring based on face count
        complexity = min(10, max(1, int(total_faces / 10) + 3))
        
        # Advanced feature detection
        holes = detect_holes(shape)
        grooves = detect_grooves(shape, cylindrical_faces)
        flat_surfaces = detect_flat_surfaces_detailed(shape)
        primary_dims = calculate_principal_dimensions(shape, (xmin, ymin, zmin, xmax, ymax, zmax), is_cylindrical)
        
        # Tessellate shape to extract mesh data
        # Default quality=0.999 for professional CAD viewer quality (0.011mm deflection)
        # This ensures perfectly smooth internal curved surfaces with no visible faceting
        quality = float(request.form.get('quality', 0.999))  # 0-1, higher = finer mesh
        mesh_data = tessellate_shape(shape, quality)
        
        # Debug: Verify feature edges are being extracted
        logger.info(f"📐 Mesh data includes {len(mesh_data.get('feature_edges', []))} feature edges")
        if mesh_data.get('feature_edges'):
            total_segments = sum(len(edge) - 1 for edge in mesh_data['feature_edges'])
            logger.info(f"   Total edge segments: {total_segments}")
        
        # Build geometry descriptor for routing selection
        geometry_descriptor = {
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "bounding_box": [width_mm, height_mm, depth_mm],
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat_surfaces,
            "holes_count": len(holes),
            "grooves_count": len(grooves),
            "complexity_score": complexity,
            "tolerance": tolerance
        }
        
        # ===== INDUSTRIAL ROUTING SELECTION =====
        routing_result = select_routings_industrial(geometry_descriptor, material)
        
        # ===== MACHINING TIME & COST ESTIMATION =====
        machining_estimate = estimate_machining_time_and_cost(
            geometry_descriptor,
            material,
            routing_result["recommended_routings"]
        )
        
        # Build feature tree for advanced feature display
        feature_tree = buildFeatureTree(holes, grooves, flat_surfaces, primary_dims, is_cylindrical, width_mm, height_mm, depth_mm)
        
        # Build enhanced result
        result = {
            'volume_cm3': round(volume_mm3 / 1000, 2),
            'surface_area_cm2': round(surface_area_mm2 / 100, 2),
            'part_width_cm': round(width_mm / 10, 2),
            'part_height_cm': round(height_mm / 10, 2),
            'part_depth_cm': round(depth_mm / 10, 2),
            'complexity_score': complexity,
            'is_cylindrical': is_cylindrical,
            'has_flat_surfaces': has_flat_surfaces,
            'cylindrical_faces': cylindrical_faces,
            'planar_faces': planar_faces,
            'total_faces': total_faces,
            'confidence': 0.95,
            'method': 'pythonOCC_geometry_parsing',
            'detected_features': {
                'holes': holes,
                'grooves': grooves,
                'flat_surfaces': flat_surfaces,
                'primary_dimensions': primary_dims
            },
            'feature_tree': feature_tree,
            'mesh_data': mesh_data,
            # Industrial routing data
            'recommended_routings': routing_result["recommended_routings"],
            'routing_reasoning': routing_result["reasoning"],
            'machining_summary': machining_estimate["machining_summary"],
            'estimated_total_cost_usd': machining_estimate["total_cost_usd"]
        }
        
        # Debug: Verify feature edges in final response
        logger.info(f"✅ Response includes mesh_data with {len(result['mesh_data'].get('feature_edges', []))} feature edges")
        
        logger.info(f"Analysis complete: cylindrical={is_cylindrical}, complexity={complexity}, holes={len(holes)}, routings={routing_result['recommended_routings']}, est_cost=${machining_estimate['total_cost_usd']}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

def detect_holes(shape):
    """Detect cylindrical holes (voids) - not outer cylindrical features"""
    holes = []
    
    try:
        # Find cylindrical faces (not edges, to avoid false positives)
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        cylindrical_faces = []
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface = BRepAdaptor_Surface(face)
            
            # Only consider cylindrical surfaces
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                axis = cylinder.Axis()
                center = axis.Location()
                direction = axis.Direction()
                radius = cylinder.Radius()
                
                # Get face properties to determine if it's a hole
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                
                cylindrical_faces.append({
                    'center': (center.X(), center.Y(), center.Z()),
                    'axis': (direction.X(), direction.Y(), direction.Z()),
                    'radius': radius,
                    'area': area,
                    'face': face
                })
            
            face_explorer.Next()
        
        # Filter to only actual holes (small radius relative to part size)
        # Get bounding box to determine part size
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        max_dimension = max(xmax - xmin, ymax - ymin, zmax - zmin)
        
        # Group cylindrical faces by axis and position
        processed = set()
        for i, cyl1 in enumerate(cylindrical_faces):
            if i in processed:
                continue
            
            # Only consider features smaller than 1/3 of the part size as potential holes
            if cyl1['radius'] > max_dimension / 3:
                processed.add(i)
                continue
            
            hole_faces = [cyl1]
            
            # Find paired cylindrical faces (same axis, similar radius)
            for j, cyl2 in enumerate(cylindrical_faces):
                if j <= i or j in processed:
                    continue
                
                # Check if axes are parallel
                axis_dot = sum(a*b for a,b in zip(cyl1['axis'], cyl2['axis']))
                if abs(abs(axis_dot) - 1.0) < 0.1:  # Nearly parallel
                    # Check if radii are similar (within 10%)
                    if abs(cyl1['radius'] - cyl2['radius']) < cyl1['radius'] * 0.1:
                        # Check if centers are aligned on axis
                        center_diff = tuple(c2-c1 for c1,c2 in zip(cyl1['center'], cyl2['center']))
                        cross_mag = sum((center_diff[i]*cyl1['axis'][j] - center_diff[j]*cyl1['axis'][i])**2 
                                       for i,j in [(0,1), (1,2), (0,2)])
                        
                        if cross_mag < cyl1['radius']**2:  # Centers aligned within radius
                            hole_faces.append(cyl2)
                            processed.add(j)
            
            processed.add(i)
            
            # Only report as hole if we have paired faces (through hole) or single small cylinder
            if len(hole_faces) >= 1:
                avg_radius = sum(f['radius'] for f in hole_faces) / len(hole_faces)
                
                # Calculate depth
                if len(hole_faces) >= 2:
                    centers = [f['center'] for f in hole_faces]
                    depth = max(math.sqrt(sum((c1[i]-c2[i])**2 for i in range(3))) 
                               for c1 in centers for c2 in centers)
                    through = depth > avg_radius * 3  # Through if depth > 1.5 * diameter
                else:
                    # Single face - likely blind hole, estimate depth from area
                    depth = hole_faces[0]['area'] / (2 * 3.14159 * avg_radius)
                    through = False
                
                # Only add if it looks like an actual hole (small relative to part)
                if avg_radius < max_dimension / 4:  # Hole radius < 25% of part size
                    # Classify orientation
                    axis = hole_faces[0]['axis']
                    abs_axis = tuple(abs(a) for a in axis)
                    max_component = max(abs_axis)
                    
                    if abs(abs_axis[2] - max_component) < 0.1:
                        orientation = 'Top+0°' if axis[2] > 0 else 'Top+180°'
                    elif abs(abs_axis[0] - max_component) < 0.1:
                        orientation = 'Right' if axis[0] > 0 else 'Left'
                    else:
                        orientation = 'Front' if axis[1] > 0 else 'Back'
                    
                    holes.append({
                        'type': 'hole',
                        'diameter_mm': round(avg_radius * 2, 2),
                        'depth_mm': round(depth, 2),
                        'through': through,
                        'position': [round(hole_faces[0]['center'][i], 2) for i in range(3)],
                        'axis': [round(axis[i], 2) for i in range(3)],
                        'orientation': orientation
                    })
    
    except Exception as e:
        logger.error(f"Error detecting holes: {e}")
    
    return holes

def detect_grooves(shape, cylindrical_face_count):
    """Detect annular grooves (parallel circular cuts)"""
    grooves = []
    
    if cylindrical_face_count < 2:
        return grooves  # Need cylindrical geometry for grooves
    
    try:
        # Look for pairs of cylindrical faces with different radii
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        cylindrical_surfaces = []
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface = BRepAdaptor_Surface(face)
            
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                axis = cylinder.Axis().Direction()
                radius = cylinder.Radius()
                
                cylindrical_surfaces.append({
                    'axis': (axis.X(), axis.Y(), axis.Z()),
                    'radius': radius
                })
            
            face_explorer.Next()
        
        # Group by similar axis and find radius differences
        if len(cylindrical_surfaces) >= 2:
            radii = sorted(set(round(s['radius'], 1) for s in cylindrical_surfaces))
            
            if len(radii) >= 2:
                # Approximate groove from radius differences
                inner_radius = radii[0]
                outer_radius = radii[-1]
                
                if outer_radius - inner_radius > 1.0:  # Significant difference
                    grooves.append({
                        'type': 'groove',
                        'inner_diameter_mm': round(inner_radius * 2, 2),
                        'outer_diameter_mm': round(outer_radius * 2, 2),
                        'depth_mm': round((outer_radius - inner_radius), 2),
                        'location': 'external',
                        'orientation': 'Radial'
                    })
    
    except Exception as e:
        logger.error(f"Error detecting grooves: {e}")
    
    return grooves

def detect_flat_surfaces_detailed(shape):
    """Detect flat surfaces with dimensions"""
    flats = []
    
    try:
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            surface = BRepAdaptor_Surface(face)
            
            if surface.GetType() == GeomAbs_Plane:
                # Get plane normal
                plane = surface.Plane()
                normal = plane.Axis().Direction()
                
                # Classify orientation
                abs_normal = tuple(abs(n) for n in [normal.X(), normal.Y(), normal.Z()])
                max_comp = max(abs_normal)
                
                if abs(abs_normal[2] - max_comp) < 0.1:
                    orientation = 'Top' if normal.Z() > 0 else 'Bottom'
                elif abs(abs_normal[0] - max_comp) < 0.1:
                    orientation = 'Side-X'
                else:
                    orientation = 'Side-Y'
                
                # Approximate area (would need more complex calculation for exact area)
                flats.append({
                    'type': 'flat',
                    'orientation': orientation,
                    'area_mm2': 100.0,  # Placeholder
                    'width_mm': 10.0,   # Placeholder
                    'length_mm': 10.0   # Placeholder
                })
            
            face_explorer.Next()
    
    except Exception as e:
        logger.error(f"Error detecting flat surfaces: {e}")
    
    return flats[:5]  # Limit to top 5 flats

def calculate_principal_dimensions(shape, bbox, is_cylindrical):
    """Calculate primary dimensions of the part"""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    
    width = xmax - xmin
    height = ymax - ymin
    depth = zmax - zmin
    
    dims = {
        'length_mm': round(depth, 2),
        'primary_axis': 'Z'
    }
    
    if is_cylindrical:
        # For cylindrical parts, identify major diameter
        dims['major_diameter_mm'] = round(max(width, height), 2)
        dims['length_mm'] = round(depth, 2)
    else:
        # For prismatic parts
        dims['width_mm'] = round(width, 2)
        dims['height_mm'] = round(height, 2)
        dims['depth_mm'] = round(depth, 2)
    
    return dims

def buildFeatureTree(holes, grooves, flat_surfaces, primary_dims, is_cylindrical, width_mm, height_mm, depth_mm):
    """
    Build a hierarchical feature tree from detected features
    
    Returns a structured tree with:
    - common_dimensions: Overall part dimensions
    - oriented_sections: Features grouped by orientation
    """
    feature_tree = {
        'common_dimensions': {},
        'oriented_sections': []
    }
    
    # Common dimensions
    if is_cylindrical:
        diameter = primary_dims.get('major_diameter_mm', max(width_mm, height_mm))
        length = primary_dims.get('length_mm', depth_mm)
        feature_tree['common_dimensions'] = {
            'type': 'cylindrical',
            'diameter_mm': round(diameter, 2),
            'length_mm': round(length, 2)
        }
    else:
        feature_tree['common_dimensions'] = {
            'type': 'prismatic',
            'width_mm': round(width_mm, 2),
            'height_mm': round(height_mm, 2),
            'depth_mm': round(depth_mm, 2)
        }
    
    # Group features by orientation
    sections = {}
    
    # Add holes to sections
    for hole in holes:
        orientation = hole.get('orientation', 'Unknown')
        if orientation not in sections:
            sections[orientation] = {
                'orientation': orientation,
                'features': []
            }
        sections[orientation]['features'].append({
            'type': 'hole',
            'diameter_mm': hole.get('diameter_mm'),
            'depth_mm': hole.get('depth_mm'),
            'through': hole.get('through', False)
        })
    
    # Add grooves to sections
    for groove in grooves:
        orientation = groove.get('orientation', 'Radial')
        if orientation not in sections:
            sections[orientation] = {
                'orientation': orientation,
                'features': []
            }
        sections[orientation]['features'].append({
            'type': 'groove',
            'inner_diameter_mm': groove.get('inner_diameter_mm'),
            'outer_diameter_mm': groove.get('outer_diameter_mm'),
            'depth_mm': groove.get('depth_mm')
        })
    
    # Add flat surfaces to sections
    for flat in flat_surfaces:
        orientation = flat.get('orientation', 'Unknown')
        if orientation not in sections:
            sections[orientation] = {
                'orientation': orientation,
                'features': []
            }
        sections[orientation]['features'].append({
            'type': 'flat_surface',
            'area_mm2': flat.get('area_mm2'),
            'width_mm': flat.get('width_mm'),
            'length_mm': flat.get('length_mm')
        })
    
    # Convert sections dict to list
    feature_tree['oriented_sections'] = list(sections.values())
    
    return feature_tree

def calculate_face_center(triangulation, transform):
    """Calculate centroid of face"""
    x_sum = y_sum = z_sum = 0
    count = triangulation.NbNodes()
    for i in range(1, count + 1):
        pnt = triangulation.Node(i)
        pnt.Transform(transform)
        x_sum += pnt.X()
        y_sum += pnt.Y()
        z_sum += pnt.Z()
    return [x_sum/count, y_sum/count, z_sum/count]

def get_average_face_normal(triangulation, transform, face_reversed):
    """Calculate average normal of face"""
    # Use first triangle's normal as representative
    triangle = triangulation.Triangle(1)
    n1, n2, n3 = triangle.Get()
    
    p1 = triangulation.Node(n1)
    p2 = triangulation.Node(n2)
    p3 = triangulation.Node(n3)
    
    p1.Transform(transform)
    p2.Transform(transform)
    p3.Transform(transform)
    
    # Cross product for normal
    edge1 = [p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()]
    edge2 = [p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()]
    
    normal = [
        edge1[1]*edge2[2] - edge1[2]*edge2[1],
        edge1[2]*edge2[0] - edge1[0]*edge2[2],
        edge1[0]*edge2[1] - edge1[1]*edge2[0]
    ]
    
    # Normalize and reverse if needed
    length = math.sqrt(sum(n*n for n in normal))
    if length > 0:
        normal = [n/length for n in normal]
    if face_reversed:
        normal = [-n for n in normal]
    
    return normal

def extract_feature_edges(shape):
    """
    Extract real CAD feature edges (B-Rep edges) for clean wireframe rendering.
    Returns list of edge polylines as 3D coordinate arrays.
    
    Returns:
        list of lists: [
            [[x1,y1,z1], [x2,y2,z2], ...],  # Edge 1 polyline
            [[x1,y1,z1], [x2,y2,z2], ...],  # Edge 2 polyline
            ...
        ]
    """
    feature_edges = []
    
    try:
        from OCC.Core import topods
        
        # Iterate through all B-Rep edges in the CAD model
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            
            # Get edge curve adapter
            curve_adapter = BRepAdaptor_Curve(edge)
            curve_type = curve_adapter.GetType()
            
            # Sample edge into polyline (adaptive sampling based on curve type)
            u_min = curve_adapter.FirstParameter()
            u_max = curve_adapter.LastParameter()
            
            # Determine sampling density based on curve type
            if curve_type == GeomAbs_Circle:
                # Circular edges: 16 samples per full circle
                arc_length = u_max - u_min
                num_samples = max(8, int(arc_length / (2 * math.pi) * 16))
            elif curve_type == 0:  # GeomAbs_Line
                # Straight lines: just endpoints
                num_samples = 2
            else:
                # BSpline curves, complex curves: 12 samples
                num_samples = 12
            
            # Sample curve
            edge_points = []
            for i in range(num_samples):
                u = u_min + (u_max - u_min) * i / (num_samples - 1)
                point = curve_adapter.Value(u)
                edge_points.append([
                    round(point.X(), 3),
                    round(point.Y(), 3),
                    round(point.Z(), 3)
                ])
            
            # Only add if edge has valid length
            if len(edge_points) >= 2:
                feature_edges.append(edge_points)
            
            edge_explorer.Next()
        
        logger.info(f"Extracted {len(feature_edges)} feature edges from CAD geometry")
        
    except Exception as e:
        logger.error(f"Error extracting feature edges: {e}")
    
    return feature_edges

def tessellate_shape(shape, quality=0.5):
    """
    Tessellate STEP shape into triangulated mesh for 3D rendering with face classification
    
    Args:
        shape: TopoDS_Shape from STEP file
        quality: 0-1 value, higher = finer mesh (more triangles, slower)
    
    Returns:
        dict with vertices, indices, normals arrays + face classifications for Meviy-style rendering
    """
    try:
        # Quality controls deflection (lower deflection = finer mesh)
        # quality=0.5   -> 0.6mm deflection (coarse, fast preview)
        # quality=0.85  -> 0.25mm deflection (good for external features)
        # quality=0.95  -> 0.15mm deflection (captures internal fillets & small radii)
        # quality=0.98  -> 0.12mm deflection (very high detail for complex internals)
        # quality=0.995 -> 0.105mm deflection (professional CAD quality)
        # quality=0.999 -> 0.011mm deflection (maximum quality, smooth internals - default)
        # quality=1.0   -> 0.01mm deflection (absolute maximum)
        
        # --- Final high-fidelity tessellation setup (logarithmic scaling) ---
        # Maps quality ∈ [0,1] to deflection in mm:
        #   0.5 → 0.5 mm   |   0.9 → 0.05 mm   |   0.99 → 0.015 mm   |   0.999 → 0.008 mm
        deflection = 0.6 * (10 ** (-(quality * 3)))  # logarithmic precision scale
        angular_deflection = 0.04  # radians (~2.3°)
        
        # Create incremental mesh with adaptive tessellation
        # Parameters: shape, linear_deflection, is_relative, angular_deflection, is_parallel
        logger.info(f"Tessellating with deflection={deflection:.4f} mm (log scale), angular={angular_deflection:.3f} rad")
        mesh = BRepMesh_IncrementalMesh(shape, deflection, False, angular_deflection, True)
        mesh.Perform()
        
        if not mesh.IsDone():
            raise Exception("Mesh tessellation failed")
        
        # Get bounding box for classification
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        max_radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
        
        # Extract triangulated geometry with face classification
        vertices = []
        indices = []
        normals = []
        face_types = []  # 'external', 'internal', 'cylindrical', 'planar'
        vertex_map = {}  # Map vertex coords to index
        current_index = 0
        
        # Iterate through all faces
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            
            if triangulation is None:
                face_explorer.Next()
                continue
            
            # Get transformation
            transform = location.Transformation()
            
            # Get face properties for classification
            surface = BRepAdaptor_Surface(face)
            face_reversed = face.Orientation() == 1  # TopAbs_REVERSED
            surface_type = surface.GetType()
            
            # Classify face type for Meviy-style color coding with normal-based detection
            face_classification = 'external'  # Default
            
            # Calculate face center and normal direction
            face_center = calculate_face_center(triangulation, transform)
            vector_to_center = [
                center_x - face_center[0],
                center_y - face_center[1],
                center_z - face_center[2]
            ]
            normal_vec = get_average_face_normal(triangulation, transform, face_reversed)
            
            # Dot product: positive = facing inward (internal), negative = facing outward
            dot_product = sum(n * v for n, v in zip(normal_vec, vector_to_center))
            # Normalize the dot product to prevent classification errors
            dot_product /= (math.sqrt(sum(v*v for v in vector_to_center)) + 1e-9)
            
            if surface_type == GeomAbs_Cylinder:
                # Cylindrical surface
                cylinder = surface.Cylinder()
                cyl_radius = cylinder.Radius()
                
                # Internal if: small radius OR normal pointing inward (normalized threshold)
                if cyl_radius < max_radius * 0.4 or dot_product > 0:
                    face_classification = 'internal'  # Internal cylindrical feature (hole)
                else:
                    face_classification = 'cylindrical'  # External cylindrical feature
                    
            elif surface_type == GeomAbs_Plane:
                # Planar surface - check if internal based on normal direction
                if dot_product > 0.5:
                    face_classification = 'internal'
                else:
                    face_classification = 'planar'
            else:
                # Other surface types (BSpline, fillets, etc.) - check if internal based on normal
                if dot_product > 0.3:
                    face_classification = 'internal'
                else:
                    face_classification = 'external'
            
            # Build local vertex map for this face
            face_vertex_start = current_index
            face_vertices = []
            
            # Extract vertices
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                # Apply transformation
                pnt.Transform(transform)
                
                coord = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
                
                if coord not in vertex_map:
                    vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])
                    vertex_map[coord] = current_index
                    face_vertices.append(current_index)
                    current_index += 1
                else:
                    face_vertices.append(vertex_map[coord])
            
            # Extract triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                
                # Map to global indices
                idx1 = face_vertices[n1 - 1]
                idx2 = face_vertices[n2 - 1]
                idx3 = face_vertices[n3 - 1]
                
                # Reverse winding if face is reversed
                if face_reversed:
                    indices.extend([idx1, idx3, idx2])
                else:
                    indices.extend([idx1, idx2, idx3])
                
                # Calculate triangle normal
                v1 = vertices[idx1*3:idx1*3+3]
                v2 = vertices[idx2*3:idx2*3+3]
                v3 = vertices[idx3*3:idx3*3+3]
                
                edge1 = [v2[i] - v1[i] for i in range(3)]
                edge2 = [v3[i] - v1[i] for i in range(3)]
                
                # Cross product
                normal = [
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0]
                ]
                
                # Normalize
                length = math.sqrt(sum(n*n for n in normal))
                if length > 0:
                    normal = [n / length for n in normal]
                
                # Ensure normals consistently point outward relative to part center
                dot_dir = sum(n * v for n, v in zip(normal, vector_to_center))
                if dot_dir < 0:
                    normal = [-n for n in normal]
                
                # Add normal for each vertex of triangle
                for _ in range(3):
                    normals.extend(normal)
                    face_types.append(face_classification)
            
            face_explorer.Next()
        
    triangle_count = len(indices) // 3

    # Extract clean feature edges from B-Rep geometry
    feature_edges = extract_feature_edges(shape)

    logger.info(f"Tessellation complete: {len(vertices)//3} vertices, {triangle_count} triangles, {len(face_types)} face classifications, {len(feature_edges)} feature edges")
    
    return {
        'vertices': vertices,
        'indices': indices,
        'normals': normals,
        'face_types': face_types,  # Face classification for each vertex
        'triangle_count': triangle_count,
        'feature_edges': feature_edges
    }
        
    except Exception as e:
        logger.error(f"Error tessellating shape: {e}")
        # Return minimal mesh on error (must include feature_edges key for consistency)
        return {
            'vertices': [],
            'indices': [],
            'normals': [],
            'face_types': [],
            'triangle_count': 0,
            'feature_edges': []
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
