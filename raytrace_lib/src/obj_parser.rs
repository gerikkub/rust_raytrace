

use crate::raytrace::{Vec3, SurfaceKind, CollisionObject, make_polygon, Triangle};

use std::fs;

struct ObjPolygon {
    corners: Vec<usize>
}

struct ObjFile {
    vertices: Vec<Vec3>,
    // norms: Vec<Vec<Vec3>>,
    faces: Vec<ObjPolygon>
}

pub struct ObjObject {
    pub objs: Vec<Triangle>
}

fn parse_obj_vertex(line: &str) -> Vec3 {
    let num_strs = line.split_whitespace();
    let parts: Vec<_> = num_strs.map(|x| x.parse::<f64>().unwrap()).collect();
    assert!(parts.len() == 3);
    Vec3(parts[0], parts[1], parts[2])
}

fn parse_obj_face(line: &str) -> ObjPolygon {
    let face_strs = line.split_whitespace();

    ObjPolygon {
        corners: face_strs.map(
                    |x| x.split('/').nth(0).unwrap().parse::<usize>().unwrap()
                    ).collect()
    }
}   

fn parse_obj_line(line: &str, ctx: &mut ObjFile) {
    if line.starts_with("v ") {
        let vec = parse_obj_vertex(line.get(2..).unwrap());
        ctx.vertices.push(vec);
    } else if line.starts_with("f ") {
        let face = parse_obj_face(line.get(2..).unwrap());
        ctx.faces.push(face);
    }
}

pub fn parse_obj(path: &str, startid: usize, offset: &Vec3, transform: (Vec3, Vec3, Vec3), surface: &SurfaceKind) -> ObjObject {

    let mut ctx = ObjFile {
        vertices: Vec::new(),
        // norms: Vec::new(),
        faces: Vec::new()
    };

    for line in fs::read_to_string(path).unwrap().lines() {
        parse_obj_line(line, &mut ctx);
    }

    let mut objs: Vec<Triangle> = Vec::new();
    let mut count = 10;

    for face in ctx.faces {
        // let corners: [Vec3; 3] = face.corners.iter().map(|p| ctx.vertices[p-1].change_basis(transform).add(offset)).collect();
        objs.push(make_polygon(
            &[ctx.vertices[face.corners[0]-1].change_basis(transform).add(offset),
              ctx.vertices[face.corners[1]-1].change_basis(transform).add(offset),
              ctx.vertices[face.corners[2]-1].change_basis(transform).add(offset)],
            surface,
            0.95,
            count + startid
        ));
        count += 1;
    }

    ObjObject {
        objs: objs
    }
}
