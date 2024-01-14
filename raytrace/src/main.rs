
use std::fs;
use std::io;
use std::io::Result;
use std::path::Path;
use std::f64::consts::{PI, FRAC_PI_3, FRAC_PI_2};

#[derive(Copy, Clone)]
pub struct Vec3(f64, f64, f64);
pub type Point = Vec3;
pub type Color = Vec3;

impl Vec3 {
    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 + other.0,
             self.1 + other.1,
             self.2 + other.2,
        )
    }

    fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 - other.0,
             self.1 - other.1,
             self.2 - other.2)
    }

    fn mult(&self, a: f64) -> Vec3 {
        Vec3(self.0 * a,
             self.1 * a, 
             self.2 * a
        )
    }

    fn len2(&self) -> f64 {
        self.0*self.0 + self.1*self.1 + self.2*self.2
    }

    fn len(&self) -> f64 {
        self.len2().sqrt()
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.0 * other.0 +
        self.1 * other.1 +
        self.2 * other.2
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3(self.1 * other.2 - self.2 * other.1,
             self.2 * other.0 - self.0 * other.2,
             self.0 * other.1 - self.1 * other.0
        )
    }

    fn unit(&self) -> Vec3 {
        let res = self.mult(1./self.len());
        res
    }

    fn orthogonal(&self) -> Vec3 {
        if self.0.abs() > 0.1 {
            Vec3(-1.*(self.1 + self.2) / self.0, 1., 1.).unit()
        } else if self.1.abs() > 0.1 {
            Vec3(1., -1.*(self.0 + self.2) / self.1, 1.).unit()
        } else if self.2.abs() > 0.1 {
            Vec3(1., 1., -1.*(self.0 + self.1) / self.2,).unit()
        } else {
            // println!("Grow: {} {} {}", self.0, self.1, self.2);
            self.unit().orthogonal()
        }
    }

    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let n = self.unit();
        let bx = n.orthogonal();
        let by = n.cross(&bx);
        (bx, by, n)
    }

    fn change_basis(&self, b: (Vec3, Vec3, Vec3)) -> Vec3 {
        Vec3(Vec3(b.0.0, b.0.1, b.0.2).dot(self),
             Vec3(b.1.0, b.1.1, b.1.2).dot(self),
             Vec3(b.2.0, b.2.1, b.2.2).dot(self)
        )
    }
}

fn make_color(color: (u8, u8, u8)) -> Color {
    Vec3((color.0 as f64) / 255.,
         (color.1 as f64) / 255.,
         (color.2 as f64) / 255.)
}

fn random_vec() -> Vec3 {
    Vec3(rand::random::<f64>() - 0.5,
         rand::random::<f64>() - 0.5,
         rand::random::<f64>() - 0.5).unit()
}

#[derive(Copy, Clone)]
pub struct Ray {
    orig: Point,
    dir: Vec3,
}

impl Ray {
    fn at(&self, t: f64) -> Point {
        Vec3(self.orig.0 + self.dir.0*t,
             self.orig.1 + self.dir.1*t,
             self.orig.2 + self.dir.2*t
        )
    }
}

/*
fn random_ray(orig: &Point, norm: &Vec3) -> Ray {

    let mut dir = random_vec();

    let cos_theta = norm.dot(&dir);

    if cos_theta < 0. {
        dir = dir.mult(-1.);
    };

    Ray {
        orig: *orig,
        dir: dir
    }
}
*/

fn reflect_ray(orig: &Point, norm: &Vec3, dir: &Vec3, fuzz: f64) -> Ray {

    let ddot = dir.dot(norm).abs();
    let dir_p = norm.mult(ddot);
    let dir_o  = dir.add(&dir_p);

    let reflect = dir_p.add(&dir_o);
    let rand_vec = random_vec().mult(fuzz);

    Ray {
        orig: *orig,
        dir: reflect.add(&rand_vec).unit()
    }
}

fn lambertian_ray(orig: &Point, norm: &Vec3) -> Ray {

    let rand_vec = random_vec();

    Ray {
        orig: *orig,
        dir: norm.add(&rand_vec)
    }
}

fn mix_color(c1: &Color, c2: &Color, a: f64) -> Color {
    c1.mult(1. - a).add(&c2.mult(a))
}

#[derive(Copy, Clone)]
pub enum SurfaceKind {
    Solid { color: Color },
    Matte { color: Color, alpha: f64},
    Reflective { scattering: f64, color: Color, alpha: f64},
}

enum CollisionFace {
    Front,
    Back,
    Side,
    Face(u64)
}

pub trait CollisionObject {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)>;
    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3;
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind;
    fn getid(&self) -> u64;
}

#[derive(Copy, Clone)]
struct Sphere {
    orig: Point,
    r: f64,
    surface: SurfaceKind,
    id: u64
}

impl CollisionObject for Sphere {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {
        let a = r.dir.dot(&r.dir);
        let cp_v = r.orig.sub(&self.orig);
        let b = r.dir.mult(2.).dot(&cp_v);
        let c = cp_v.dot(&cp_v) - self.r*self.r;

        let disc = b*b - 4.*a*c;
        if disc < 0. {
            None
        } else {
            let tp = (-1. * b + disc.sqrt()) / (2. * a);
            let tm = (-1. * b - disc.sqrt()) / (2. * a);

            if tp < 0. && tm < 0. {
                None
            } else {
                let t = if tp < 0. { tm }
                        else if tm < 0. { tp }
                        else if tm < tp { tm }
                        else { tm };
                
                let point = r.at(t);
                Some((t, point, CollisionFace::Front))
            }
        }
    }

    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3 {
        match f {
            CollisionFace::Front => p.sub(&self.orig).unit(),
            _ => panic!("Invalid face for Sphere")
        }
    }

    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match f {
            CollisionFace::Front => self.surface,
            _ => panic!("Invalid face for Sphere")
        }
    }

    fn getid(&self) -> u64 {
        self.id
    }
}

#[derive(Copy, Clone)]
struct Disk {
    orig: Point,
    norm: Vec3,
    r: f64,
    depth: f64,
    surface: SurfaceKind,
    side_surface: SurfaceKind,
    id: u64
}

impl CollisionObject for Disk {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {

        // t = -1 * (norm * (orig - Rorig)) / (norm * Rdir)
        let surf_front = self.orig.add(&self.norm.mult(self.depth));
        let t_front = self.norm.dot(&surf_front.sub(&r.orig)) / self.norm.dot(&r.dir);
        let p_front = r.at(t_front);

        let surf_back = self.orig.sub(&self.norm.mult(self.depth));
        let t_back = self.norm.dot(&surf_back.sub(&r.orig)) / self.norm.dot(&r.dir);
        let p_back = r.at(t_back);

        // Hit in negative time, ignore
        if t_back < 0. && t_front < 0. {
            return None;
        }

        let hit_front = p_front.sub(&surf_front).len() < self.r &&
                        t_front >= 0.;
        let hit_back = p_back.sub(&surf_back).len() < self.r &&
                       t_back >= 0.;
        let hit_face = hit_front || hit_back;

        if hit_front && hit_back {
            // Hit both faces. Choose the lowest positive face
            if t_front < t_back {
                Some((t_front, p_front, CollisionFace::Front))
            } else {
                Some((t_back, p_back, CollisionFace::Back))
            }
        } else {
            // hit one or no faces. This means it could go through
            // the edge of the disk

            let n_basis = self.norm.basis();
            // println!("Basis Vectors: ");
            // println!(" {} {} {}", n_basis.0.0, n_basis.0.1, n_basis.0.2);
            // println!(" {} {} {}", n_basis.1.0, n_basis.1.1, n_basis.1.2);
            // println!(" {} {} {}", n_basis.2.0, n_basis.2.1, n_basis.2.2);

            let r_n_temp = Ray {
                orig: r.orig.sub(&self.orig).change_basis(n_basis),
                dir: r.dir.change_basis(n_basis),
            };
            let r_n = Ray {
                orig: Vec3(r_n_temp.orig.0, r_n_temp.orig.1, 0.),
                dir: Vec3(r_n_temp.dir.0, r_n_temp.dir.1, 0.),
            };

            // println!("New Ray: ");
            // println!(" {} {} {}", r_n.orig.0, r_n.orig.1, r_n.orig.2);
            // println!(" {} {} {}", r_n.dir.0, r_n.dir.1, r_n.dir.2);

            let a = r_n.dir.dot(&r_n.dir);
            let b = 2. * r_n.orig.dot(&r_n.dir);
            let c = r_n.orig.dot(&r_n.orig) - self.r*self.r;

            let discriminate = b*b - 4.*a*c;

            let (hit_side, hit_side_t) =
                if discriminate >= 0. {
                    let tp = (-1. * b + discriminate.sqrt()) / (2. * a);
                    let tm = (-1. * b - discriminate.sqrt()) / (2. * a);

                    let kp = self.orig.sub(&r.at(tp)).dot(&self.norm);
                    let km = self.orig.sub(&r.at(tm)).dot(&self.norm);

                    let valid_p = tp > 0. && kp.abs() < self.depth;
                    let valid_m = tm > 0. && km.abs() < self.depth;

                    if !valid_p && !valid_m {
                        (false, 0.)
                    } else if !valid_p {
                        (true, tm)
                    } else if !valid_m {
                        (true, tp)
                    } else {
                        if tm < tp {
                            (true, tm)
                        } else {
                            (true, tp)
                        }
                    }
                } else {
                    (false, 0.)
                };

            let (hit_face, hit_face_t) =
                if hit_front {
                    (true, t_front)
                } else if hit_back {
                    (true, t_back)
                } else {
                    (false, 0.)
                };
        
            if hit_face && hit_side {
                if hit_face_t < hit_side_t {
                    Some((hit_face_t,
                        r.at(hit_face_t),
                        if hit_front { CollisionFace::Front } else { CollisionFace::Back}))
                } else {
                    Some((hit_side_t, r.at(hit_side_t), CollisionFace::Side))
                }
            } else if hit_face {
                Some((hit_face_t,
                        r.at(hit_face_t),
                        if hit_front { CollisionFace::Front } else { CollisionFace::Back}))
            } else if hit_side {
                Some((hit_side_t, r.at(hit_side_t), CollisionFace::Side))
            } else {
                None
            }
        }

        // println!("Disk at {} {} {} {}", t, p.0, p.1, p.2);

        // if t > 0. {
        //     let dist = p.sub(&self.orig).len();
        //     // println!("Disk hit {} {}", t, dist);
        //     if dist < self.r {
        //         Some((t, p))
        //     } else {
        //         None
        //     }
        // } else {
        //     None
        // }
    }

    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3 {
        // Note: This actually depends on the direction the ray came from
        match f {
            CollisionFace::Front => self.norm,
            CollisionFace::Back => self.norm.mult(-1.),
            CollisionFace::Side => {
                let k = self.orig.sub(p).dot(&self.norm);
                let p2 = p.add(&self.norm.mult(k));
                p2.sub(&self.orig).unit()
            },
            _ => panic!("Invalid face for disk")
        }
    }
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match f {
            CollisionFace::Front => self.surface,
            CollisionFace::Back => self.surface,
            CollisionFace::Side => self.side_surface,
            _ => panic!("Invalid face for disk")
        }
    }
    fn getid(&self) -> u64 {
        self.id
    }
}

struct Scene<'a> {
    objs: &'a Vec<Box<dyn CollisionObject>>
}

fn color_ray(r: &Ray, s: &Scene, obj: &Box<dyn CollisionObject>, point: &Point, face: &CollisionFace, depth: i64) -> Color {
    match obj.getsurface(face) {
        SurfaceKind::Solid {color} => {
            color
        },
        SurfaceKind::Matte {color, alpha} => {
            mix_color(&color,
                      &project_ray(&lambertian_ray(point,
                                                   &obj.normal(point, face)),
                                   s,
                                   obj.getid(),
                                   depth + 1),
                      alpha)
        },
        SurfaceKind::Reflective {scattering, color, alpha}  => {
            mix_color(&color,
                      &project_ray(&reflect_ray(point,
                                                &obj.normal(point, face),
                                                &r.dir,
                                                scattering),
                                   s,
                                   obj.getid(),
                                   depth + 1),
                       alpha)
        }
    }
}

fn project_ray(r: &Ray, s: &Scene, ignore_objid: u64, depth: i64) -> Color {

    if depth > 5 {
        return Vec3(0., 0., 0.);
    }

    let blue = Vec3(0.5, 0.7, 1.);

    let intersections: Vec<(f64, Point, CollisionFace, &Box<dyn CollisionObject>)> = s.objs.iter().filter_map(
        |s| {
            if ignore_objid == s.getid() {
                None
            } else {
                match s.intersects(&r) {
                    Some(p) => Some((p.0, p.1, p.2, s)),
                    None    => None
                }
            }
        }).collect();

    if intersections.len() == 0 {
        if r.dir.2 > 0. {
            blue
        } else {
            blue
        }
    } else {
        let (_dist, point, face, obj) = intersections.iter().fold(&intersections[0],
            |acc, x| {
                let (dist, _, _, _) = x;
                let (acc_dist, _, _, _) = acc;
                if dist < acc_dist { x } else { acc }
            });
            color_ray(r, s, obj, point, &face, depth)
    }
}

struct Viewport {
    width: u32,
    height: u32,

    orig: Point,
    cam: Point,
    
    vu: Vec3,
    vv: Vec3,

    samples_per_pixel: u32
}

fn create_viewport(px: (u32, u32), size: (f64, f64), pos: &Point, dir: &Vec3, fov: f64, c_roll: f64) -> Viewport {

    let dist = size.0 / (2. * (fov.to_radians() / 2.).tan());

    let cam = *pos;

    let roll = -1.*(-1. * dir.1).atan2(dir.2);
    let pitch = -1.*dir.0.asin();
    let yaw = -1.*c_roll;

    println!("YPR: {} {} {}", yaw, pitch, roll);

    let rot_basis = (
        Vec3(yaw.cos()*pitch.cos(),
             yaw.sin()*pitch.cos(),
             -1.*pitch.sin()),

        Vec3(yaw.cos()*pitch.sin()*roll.sin() - yaw.sin()*roll.cos(),
             yaw.sin()*pitch.sin()*roll.sin() + yaw.cos()*roll.cos(),
             pitch.cos()*roll.sin()),

        Vec3(yaw.cos()*pitch.sin()*roll.cos() + yaw.sin()*roll.sin(),
             yaw.sin()*pitch.sin()*roll.cos() - yaw.cos()*roll.sin(),
             pitch.cos()*roll.cos())
    );

    println!("Basis:");
    println!(" {} {} {}", rot_basis.0.0, rot_basis.0.1, rot_basis.0.2);
    println!(" {} {} {}", rot_basis.1.0, rot_basis.1.1, rot_basis.1.2);
    println!(" {} {} {}", rot_basis.2.0, rot_basis.2.1, rot_basis.2.2);

    let orig = Vec3(1.*size.1/2., -1.*size.0/2., dist);
    let orig_r = cam.add(&orig.change_basis(rot_basis));

    println!("orig: {} {} {}", orig.0, orig.1, orig.2);
    println!("orig_r: {} {} {}", orig_r.0, orig_r.1, orig_r.2);

    let vu = Vec3(0., size.0, 0.);
    let vu_r = vu.change_basis(rot_basis);

    let vv = Vec3(-1. * size.1, 0., 0.);
    let vv_r = vv.change_basis(rot_basis);

    println!("vu_r: {} {} {}", vu_r.0, vu_r.1, vu_r.2);
    println!("vv_r: {} {} {}", vv_r.0, vv_r.1, vv_r.2);

    Viewport {
        width: px.0,
        height: px.1,
        orig: orig_r,
        cam: cam,
        vu: vu_r,
        vv: vv_r,
        samples_per_pixel: 100
    }
}

impl Viewport {

    fn pixel_ray(&self, px: (u32, u32)) -> Ray {

        let px_x = px.0 as f64;
        let px_y = px.1 as f64;

        let vu_delta = self.vu.mult(1. / self.width as f64);
        let vv_delta = self.vv.mult(1. / self.height as f64);

        let u_off: f64 = rand::random();
        let v_off: f64 = rand::random();

        let vu_frac = vu_delta.mult(px_y + u_off);
        let vv_frac = vv_delta.mult(px_x + v_off);

        let px_u = self.orig.add(&vu_frac).add(&vv_frac);

        Ray {
            orig: px_u,
            dir: px_u.sub(&self.cam).unit()
        }
    }

    fn walk_rays(&self, s: &Scene, data: & mut[Color]) {
        for x in 0..self.height {
            for y in 0..self.width {
                let mut acc = Vec3(0., 0., 0.);
                for _i in 0..self.samples_per_pixel {
                    let ray_color = project_ray(&self.pixel_ray((x,y)), s, 0, 0);
                    acc = acc.add(&ray_color);
                }
                data[(x*self.width + y) as usize] = acc.mult(1./(self.samples_per_pixel as f64));
            }
        }
    }
}


fn write_png(f: fs::File, img_size: (u32, u32), data: &[Color]) -> Result<()> {

    let ref mut w = io::BufWriter::new(f);
    let mut encoder = png::Encoder::new(w, img_size.0, img_size.1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data_int = Vec::with_capacity(data.len() * 3);
    for c in data {
        data_int.push((c.0 * 255.) as u8);
        data_int.push((c.1 * 255.) as u8);
        data_int.push((c.2 * 255.) as u8);
    }

    writer.write_image_data(&data_int).unwrap();

    Ok(())
}

fn main() -> Result<()> {
    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    let aspect = 480. / 640.;
    let width = 640;
    let height = 480;

    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let v = create_viewport((width, height),
                            (1., 1. * aspect),
                            // &Vec3(0.8, 0., 3.),
                            // &Vec3(-1., 0.0, 3.).unit(),
                            &Vec3(0.0, 0., 2.),
                            &Vec3(0., 0.0, 1.).unit(),
                            90.,
                            0_f64.to_radians());

    // let v2 = create_viewport((width, height),
    //                         (1., 1. * aspect),
    //                         // &Vec3(5., 0., 5.),
    //                         // &Vec3(-1., 0.0, 0.).unit(),
    //                         &Vec3(2., 0., 0.),
    //                         &Vec3(-1., 0.0, 1.).unit(),
    //                         140.,
    //                         0_f64.to_radians());
    // return Ok(());

    let mut objs: Vec<Box<dyn CollisionObject>> = Vec::new();
    objs.push(Box::new(Sphere {
        // orig: Vec3(-15.5 + (15.*15. - 1.*1. as f64).sqrt(),
        //            0.,
        //            6. - 1.),
        orig: Vec3(-0.2,
                   0.,
                   5.),
        r: 0.5,
        // surface: SurfaceKind::Solid {
        //     color: Vec3(1., 0.3, 0.3),
        //     // alpha: 0.5
        // },
        // surface: SurfaceKind::Matte {
        //     color: Vec3(0.3, 0.3, 0.3),
        //     alpha: 0.5
        // },
        surface: SurfaceKind::Reflective {
            scattering: 0.1,
            color: Vec3(0.8, 0.3, 0.3),
            alpha: 0.5
        },
        id: 1
    }));
    objs.push(Box::new(Sphere {
        orig: Vec3(-50.5, 0., 5.),
        r: 50.,
        surface: SurfaceKind::Matte {
            color: make_color((0, 150, 70)),
            alpha: 0.4
        },
        id: 2
    }));
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

    let mut sphere_objs: Vec<Sphere> = Vec::new();
    'build_spheres: for i in 0..70 {
        let temp = random_vec().mult(1.5).add(&Vec3(0., 0., 5.));
        let orig = Vec3(-0.5, temp.1, temp.2);
        for obj in &sphere_objs {
            let dist = orig.sub(&obj.orig).len();
            if dist < 0.2 {
                continue 'build_spheres;
            }
        }
        let temp2 = random_vec();
        let sphere = Sphere {
            orig: orig,
            r: 0.1,
            // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
            surface: SurfaceKind::Matte {
                color: Vec3(temp2.0.abs(), temp2.1.abs(), temp2.2.abs()),
                alpha: (rand::random::<f64>() / 2.)
            },
            id: 5 + i
        };
        sphere_objs.push(sphere);
        objs.push(Box::new(sphere));
    }

    objs.push(Box::new(Disk {
        // orig: Vec3(0.5, 0., 7.),
        // norm: Vec3(-1., 0., -1.).unit(),
        orig: Vec3(1., 0.0, 5.5),
        norm: Vec3(-0.8, 0.2, -0.5).unit(),
        r: 1.5,
        depth: 0.05,
        surface: SurfaceKind::Reflective {
            scattering: 0.05,
            color: make_color((200, 60, 90)),
            alpha: 0.9
        },
        side_surface: SurfaceKind::Matte {
            color: make_color((30, 30, 30)),
            alpha: 0.3
        },
        id: 5
    }));


    let s = Scene {
        objs: &objs
    };

    v.walk_rays(&s, &mut data);

    write_png(file, (width, height), &data)
}
