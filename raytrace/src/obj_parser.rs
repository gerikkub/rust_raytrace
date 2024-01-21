

use crate::raytrace::{Vec3, SurfaceKind, CollisionObject,
                      make_triangle, make_color, Triangle};

use std::fs;

struct ObjFile {
    vertices: Vec<Vec3>,
    faces: Vec<(usize, usize, usize)>
}

pub struct ObjObject {
    pub objs: Vec<Triangle>
}

fn parse_obj_vertex(line: &str) -> Vec3 {
    let num_strs = line.split_whitespace();
    let nums: Vec<_> = num_strs.map(|x| x.parse::<f64>().unwrap()).collect();
    assert!(nums.len() == 3);
    Vec3(nums[0], nums[1], nums[2])
}

fn parse_obj_face(line: &str) -> (usize, usize, usize) {
    let vert_strs = line.split_whitespace();
    let verts: Vec<_> = vert_strs.map(
            |x| x.split('/').nth(0).unwrap().parse::<usize>().unwrap()
        ).collect();
    assert!(verts.len() == 3);
    (verts[0], verts[1], verts[2])
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

pub fn parse_obj(path: &str, offset: &Vec3, transform: (Vec3, Vec3, Vec3)) -> ObjObject {

    let mut ctx = ObjFile {
        vertices: Vec::new(),
        faces: Vec::new()
    };

    for line in fs::read_to_string(path).unwrap().lines() {
        parse_obj_line(line, &mut ctx);
    }

    let mut objs: Vec<Triangle> = Vec::new();
    let mut count = 10;

    for face in ctx.faces {
        objs.push(make_triangle(
            (
                ctx.vertices[face.0 - 1].change_basis(transform).add(offset),
                ctx.vertices[face.1 - 1].change_basis(transform).add(offset),
                ctx.vertices[face.2 - 1].change_basis(transform).add(offset)
            ),
            &SurfaceKind::Solid { color: make_color((150, 150, 150)) },
            // &SurfaceKind::Matte { 
            //     color: make_color((150, 150, 150)),
            //     alpha: 0.4
            // },
            // &SurfaceKind::Reflective {
            //     scattering: 0.3,
            //     color: make_color((200, 200, 200)),
            //     alpha: 0.9
            // },
            count
        ));
        count += 1;
    }

    ObjObject {
        objs: objs
    }
}