
pub mod raytrace;
use raytrace::{Vec3, Sphere, Disk, Scene, SurfaceKind, CollisionObject, create_transform};

pub mod obj_parser;

use std::fs;
use std::io::Result;
use std::path::Path;

fn main() -> Result<()> {
    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    // let aspect = 1440. / 2560.;
    // let width = 2560;
    // let height = 1440;

    let aspect = 480. / 640.;
    // let width = 160;
    // let height = 120;
    let width = 640;
    let height = 480;

    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let obj_data = obj_parser::parse_obj("teapot.obj",
                                         &Vec3(-0.5, 0., 5.),
                                         create_transform(&Vec3(0., 0., 1.),
                                                          270_f64.to_radians()));
                                        //  create_transform(&Vec3(1., 1., 0.5), 0.));


    println!("Found {} objects", obj_data.objs.len());

    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            // &Vec3(0.8, 0., 3.),
                            // &Vec3(-1., 0.0, 3.).unit(),
                            &Vec3(2.5, 0., 0.),
                            &Vec3(-0.4, 0.0, 1.).unit(),
                            80.,
                            0_f64.to_radians(),
                            15);
    
    let mut scene_objs = obj_data.objs;

    scene_objs.push(CollisionObject::Sphere(Sphere {
        orig: Vec3(-50.5, 0., 5.),
        r: 50.,
        surface: SurfaceKind::Matte {
            color: raytrace::make_color((40, 40, 40)),
            alpha: 0.4
        },
        id: 2
    }));

    let s = Scene {
        objs: &scene_objs
    };

    v.walk_rays(&s, &mut data, 15);

    raytrace::write_png(file, (width, height), &data);
    return Ok(());

    // let v2 = create_viewport((width, height),
    //                         (1., 1. * aspect),
    //                         // &Vec3(5., 0., 5.),
    //                         // &Vec3(-1., 0.0, 0.).unit(),
    //                         &Vec3(2., 0., 0.),
    //                         &Vec3(-1., 0.0, 1.).unit(),
    //                         140.,
    //                         0_f64.to_radians());
    // return Ok(());

    // let mut objs: Vec<Box<dyn raytrace::CollisionObject>> = Vec::new();
    // objs.push(Box::new(Sphere {
    //     // orig: Vec3(-15.5 + (15.*15. - 1.*1. as f64).sqrt(),
    //     //            0.,
    //     //            6. - 1.),
    //     orig: Vec3(-0.2,
    //                0.,
    //                5.),
    //     r: 0.5,
    //     // surface: SurfaceKind::Solid {
    //     //     color: Vec3(1., 0.3, 0.3),
    //     //     // alpha: 0.5
    //     // },
    //     // surface: SurfaceKind::Matte {
    //     //     color: Vec3(0.3, 0.3, 0.3),
    //     //     alpha: 0.5
    //     // },
    //     surface: SurfaceKind::Reflective {
    //         scattering: 0.1,
    //         color: Vec3(0.8, 0.3, 0.3),
    //         alpha: 0.5
    //     },
    //     id: 1
    // }));

    // objs.push(Box::new(raytrace::make_triangle(
    //           (Vec3(-0.3, 0., 4.),
    //             Vec3(0., 0., 6.),
    //             Vec3(0.5, 2., 5.)
    //           ),
    //           &SurfaceKind::Matte {
    //             color: raytrace::make_color((200, 50, 50)),
    //             alpha: 0.3
    //         },
    //         10)));

    // objs.push(Box::new(Sphere {
    //     orig: Vec3(0.5, 0.0, 5.8),
    //     r: 0.2,
    //     // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
    //     // surface: SurfaceKind::Reflective(0.1),
    //     surface: SurfaceKind::Reflective {
    //         scattering: 0.1,
    //         color: make_color((51, 255, 165)),
    //         alpha: 0.7
    //     },
    //     id: 3
    // }));
    // objs.push(Box::new(Sphere {
    //     orig: Vec3(0., -1.2, 6.2),
    //     r: 0.5,
    //     // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
    //     surface: SurfaceKind::Reflective {
    //         scattering: 0.3,
    //         color: make_color((255, 87, 51)),
    //         alpha: 0.5
    //     },
    //     id: 4
    // }));

    // let mut sphere_objs: Vec<Sphere> = Vec::new();
    // 'build_spheres: for i in 0..70 {
    //     let temp = random_vec().mult(1.5).add(&Vec3(0., 0., 5.));
    //     let orig = Vec3(-0.5, temp.1, temp.2);
    //     for obj in &sphere_objs {
    //         let dist = orig.sub(&obj.orig).len();
    //         if dist < 0.2 {
    //             continue 'build_spheres;
    //         }
    //     }
    //     let temp2 = random_vec();
    //     let sphere = Sphere {
    //         orig: orig,
    //         r: 0.1,
    //         // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
    //         surface: SurfaceKind::Matte {
    //             color: Vec3(temp2.0.abs(), temp2.1.abs(), temp2.2.abs()),
    //             alpha: (rand::random::<f64>() / 2.)
    //         },
    //         id: 5 + i
    //     };
    //     sphere_objs.push(sphere);
    //     objs.push(Box::new(sphere));
    // }

    // objs.push(CollisionObject::Dist(Disk {
    //     // orig: Vec3(0.5, 0., 7.),
    //     // norm: Vec3(-1., 0., -1.).unit(),
    //     orig: Vec3(1., 0.0, 5.5),
    //     norm: Vec3(-0.8, 0.2, -0.5).unit(),
    //     r: 1.5,
    //     depth: 0.05,
    //     surface: SurfaceKind::Reflective {
    //         scattering: 0.02,
    //         color: raytrace::make_color((200, 60, 90)),
    //         alpha: 0.9
    //     },
    //     side_surface: SurfaceKind::Matte {
    //         color: raytrace::make_color((30, 30, 30)),
    //         alpha: 0.3
    //     },
    //     id: 5
    // }));


    // let s = Scene {
    //     objs: &objs
    // };

    // v.walk_rays(&s, &mut data, 8);

    // raytrace::write_png(file, (width, height), &data)
}