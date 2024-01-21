
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
    // let width = 160;
    // let height = 120;
    // let width = 640;
    // let height = 480;

    // let aspect = 1.;
    // let width = 1;
    // let height = 1;

    // let tri = make_triangle((Vec3(-0.9, -1., -1.1),
    //                          Vec3(-0.9, -1.3, -0.8),
    //                          Vec3(-1.1, -0.7, -0.8)),
    //                          &SurfaceKind::Solid { color: make_color((0, 0, 0)) },
    //                          0);
    // let tri = make_triangle((Vec3(-1.9, -0., -1.2),
    //                          Vec3(-1.1, -1.7, 0.),
    //                          Vec3(-0.2, -1.3, -1.1)),
    //                          &SurfaceKind::Solid { color: make_color((0, 0, 0)) },
    //                          0);
    // println!("{:?}", tri);
    // let a = raytrace::box_contains_triangle(&Vec3(0., 0., 0.), 1., &tri);
    // println!("{}", a);
    // return Ok(());


    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let obj_data = obj_parser::parse_obj("teapot.obj",
                                         &Vec3(0., 0.5, 5.),
                                         raytrace::create_transform(&Vec3(0., 0., 1.),
                                                                    270_f64.to_radians()));
                                        //  create_transform(&Vec3(1., 1., 0.5), 0.));


    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &Vec3(3., 0., 0.),
                            &Vec3(-0.37, 0.3, 1.).unit(),
                            100.,
                            // &Vec3(-0.37, 0.3, 1.).unit(),
                            // 1.,
                            0_f64.to_radians(),
                            100);
    
    println!("Viewport: {:?}", v);
    
    let bbox = raytrace::build_bounding_box(&obj_data.objs,
                                            &Vec3(0., 0., 0.),
                                            20.,
                                            20,
                                            10);

    // let t = &obj_data.objs.iter().find(|x| x.id == 1995).unwrap();
    // println!("");
    // println!("{:?}", t);

    // let r = raytrace::Ray {
    //     orig: Vec3(-17.638537549881196, 15.517733148552326, 51.72577716184108),
    //     dir: Vec3(-0.33403897381257164, 0.27084241119938246, 0.9028080373312748)
    // };

    // let bboxes = bbox.find_obj(1995);
    // for b in bboxes {
    //     b.print();
    //     println!("{}", b.collides(&r));
    // }

    // println!("{}", t.intersects(&r).is_some());

    // bbox.print_tree();
    // return Ok(());

    let s = Scene {
        boxes: bbox
    };

    let progress_ctx = v.walk_rays(&s, &mut data, 20);

    progress_ctx.print_stats();

    let _ = raytrace::write_png(file, (width, height), &data);
    return Ok(());
}