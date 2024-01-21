
pub mod raytrace;
use raytrace::{Vec3, Sphere, Disk, Scene, SurfaceKind, CollisionObject, BoundingBox, make_triangle, make_color};
use crate::raytrace::Collidable;

pub mod obj_parser;
pub mod progress;

use std::fs;
use std::io::Result;
use std::path::Path;

fn main() -> Result<()> {
    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    let aspect = 1440. / 2560.;
    let width = 2560;
    let height = 1440;

    // let aspect = 480. / 640.;
    // let width = 640;
    // let height = 480;

    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let obj_data = obj_parser::parse_obj("teapot.obj", 10,
                                         &Vec3(0., 0.5, 5.),
                                         raytrace::create_transform(&Vec3(0., 0.3, 1.).unit(),
                                                                    270_f64.to_radians()),
                                         &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                                               alpha: 0.2 });


    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &Vec3(4., -0.5, 0.),
                            &Vec3(-0.37, 0.3, 1.).unit(),
                            80.,
                            0_f64.to_radians(),
                            10);
    
    println!("Viewport: {:?}", v);
    
    let bbox = raytrace::build_bounding_box(&obj_data.objs,
                                            &Vec3(0., 0., 0.),
                                            20.,
                                            20,
                                            10);

    let mut otherobjs: Vec<CollisionObject> = Vec::new();


    otherobjs.push(CollisionObject::Disk(Disk {
        orig: Vec3(4., 4., 7.),
        norm: Vec3(-0.3, -0.55, -0.5).unit(),
        r: 2.,
        depth: 0.1,
        surface: SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
        side_surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
        id: 1
    }));

    otherobjs.push(CollisionObject::Sphere(Sphere {
        orig: Vec3(-50., 0., 5.),
        r: 50.,
        surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                      alpha: 0.2 },
        id: 2
    }));
        
    let s = Scene {
        boxes: bbox,
        otherobjs: otherobjs
    };

    let progress_ctx = v.walk_rays(&s, &mut data, 20);

    progress_ctx.print_stats();

    let _ = raytrace::write_png(file, (width, height), &data);
    return Ok(());
}