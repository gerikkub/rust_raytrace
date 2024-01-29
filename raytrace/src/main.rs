

use raytrace_lib::raytrace::{Vec3, Sphere, Disk, Scene, SurfaceKind, CollisionObject, make_color, LightSource};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;

// pub mod obj_parser;
// pub mod progress;

use std::fs;
use std::io::Result;
use std::path::Path;


fn main() -> Result<()> {
    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    // let aspect = 2160. / 3840.;
    // let width = 3840;
    // let height = 2160;

    // let aspect = 1440. / 2560.;
    // let width = 2560;
    // let height = 1440;

    let aspect = 480. / 640.;
    let width = 640;
    let height = 480;

    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let obj_data = obj_parser::parse_obj("teapot_tri.obj", 10,
                                         &Vec3(0., 0.5, 5.),
                                         raytrace::create_transform(&Vec3(0., 0.3, 1.).unit(),
                                                                    270_f64.to_radians()),
                                         &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                                               alpha: 0.2 });


    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &Vec3(2., 0., 0.),
                            &Vec3(0., 0., 1.).unit(),
                            90.,
                            0_f64.to_radians(),
                            2,
                            10);
    
    println!("Viewport: {:?}", v);
    
    // let bbox = raytrace::build_empty_box();
    // let bbox = raytrace::build_trivial_bounding_box(&obj_data.objs,
    //                                         &Vec3(0., 0., 0.),
    //                                         20.);
    let bbox = raytrace::build_bounding_box(&obj_data.objs,
                                            &Vec3(0., 0., 0.),
                                            20.,
                                            20,
                                            15);

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

    otherobjs.push(CollisionObject::Disk(Disk {
        orig: Vec3(4., -3., 5.),
        norm: Vec3(-0.5, 1.0, -0.5).unit(),
        r: 1.,
        depth: 0.04,
        surface: SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
        side_surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
        id: 3
    }));

    // let poly = make_polygon(&vec![Vec3(1., -1., 3.),
    //                         Vec3(1., 1., 3.),
    //                         Vec3(3., 1., 3.),
    //                         Vec3(3., -1., 3.)],
    //                         &SurfaceKind::Solid { color: make_color((220, 20, 20)) },
    //                         0.99, 3);

    // otherobjs.push(CollisionObject::ConvexPolygon(poly));

    // otherobjs.push(CollisionObject::Triangle(make_triangle((Vec3(1., 1.5, 3.5),
    //                                                         Vec3(1., 3., 4.),
    //                                                         Vec3(3., 1.5, 4.)),
    //                                                        &SurfaceKind::Solid { color: make_color((22, 220, 20)) },
    //                                                        0.99, 4)));

    let light_orig = Vec3(7., 15., 4.);

    // otherobjs.push(CollisionObject::Sphere(Sphere {
    //     orig: light_orig,
    //     r: 1.0,
    //     surface: SurfaceKind::Solid { color: make_color((255, 255, 255)) },
    //     id: 3
    // }));

    let light = LightSource {
        orig: light_orig,
        len2: 0.01
    };

    let s = Scene {
        boxes: bbox,
        otherobjs: otherobjs,
        lights:  Some(light)
        // lights:  None
    };

    let progress_ctx = v.walk_rays(&s, &mut data, 12);

    progress_ctx.print_stats();

    let _ = raytrace::write_png(file, (width, height), &data);
    return Ok(());
}