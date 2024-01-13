
use std::fs;
use std::io;
use std::io::Result;
use std::path::Path;
use std::f64::consts::{PI, FRAC_PI_3, FRAC_PI_2};

#[derive(Copy, Clone)]
struct Vec3(f64, f64, f64);
type Point = Vec3;
type Color = Vec3;

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
}

fn make_color(color: (u8, u8, u8)) -> Color {
    Vec3((color.0 as f64) / 255.,
         (color.1 as f64) / 255.,
         (color.2 as f64) / 255.)
}

#[derive(Copy, Clone)]
struct Ray {
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

fn random_ray(orig: &Point, norm: &Vec3) -> Ray {

    let mut dir = Vec3(0., 0., 0.);
    dir = Vec3((rand::random::<f64>() * 2.) - 1.,
               (rand::random::<f64>() * 2.) - 1.,
               (rand::random::<f64>() * 2.) - 1.);
    dir = dir.unit();

    let cos_theta = norm.dot(&dir);

    if cos_theta < 0. {
        dir = dir.mult(-1.);
    };

    Ray {
        orig: *orig,
        dir: dir
    }
}

fn reflect_ray(orig: &Point, norm: &Vec3, dir: &Vec3, fuzz: f64) -> Ray {

    let ddot = dir.dot(norm).abs();
    let dir_p = norm.mult(ddot);
    let dir_o  = dir.add(&dir_p);

    let reflect = dir_p.add(&dir_o);
    let rand_vec = Vec3(rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5).unit().mult(fuzz);

    Ray {
        orig: *orig,
        dir: reflect.add(&rand_vec).unit()
    }
}

fn lambertian_ray(orig: &Point, norm: &Vec3) -> Ray {

    let rand_vec = Vec3(rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5,
                        rand::random::<f64>() - 0.5).unit();

    Ray {
        orig: *orig,
        dir: norm.add(&rand_vec)
    }
}

fn mix_color(c1: &Color, c2: &Color, a: f64) -> Color {
    c1.mult(1. - a).add(&c2.mult(a))
}

#[derive(Copy, Clone)]
enum SurfaceKind {
    Solid { color: Color },
    Matte { color: Color, alpha: f64},
    Reflective { scattering: f64, color: Color, alpha: f64},
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

fn create_viewport(px: (u32, u32), size: (f64, f64)) -> Viewport {
    Viewport {
        width: px.0,
        height: px.1,
        orig: Vec3(1.*size.1/2., -1.*size.0/2., 1.),
        cam: Vec3(0., 0., -1.),
        vu: Vec3(0., size.0, 0.),
        vv: Vec3(-1. * size.1, 0., 0.),
        samples_per_pixel: 100
    }
}

pub trait CollisionObject {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point)>;
    fn normal(&self, p: &Point) -> Vec3;
    fn getsurface(&self) -> SurfaceKind;
    fn getid(&self) -> u64;
}

struct Sphere {
    orig: Point,
    r: f64,
    surface: SurfaceKind,
    id: u64
}

impl CollisionObject for Sphere {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point)> {
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
                let dist = self.orig.sub(&point).len2();
                Some((t, point))
            }
        }
    }

    fn normal(&self, p: &Point) -> Vec3 {
        p.sub(&self.orig).unit()
    }

    fn getsurface(&self) -> SurfaceKind {
        self.surface
    }

    fn getid(&self) -> u64 {
        self.id
    }
}

struct Scene<'a> {
    aspect: f64,
    objs: &'a Vec<Box<dyn CollisionObject>>
}

fn color_ray(r: &Ray, s: &Scene, obj: &Box<dyn CollisionObject>, point: &Point, depth: i64) -> Color {
    match obj.getsurface() {
        SurfaceKind::Solid {color} => {
            color
        },
        SurfaceKind::Matte {color, alpha} => {
            mix_color(&color,
                      &project_ray(&lambertian_ray(point,
                                                   &obj.normal(point)),
                                   s,
                                   obj.getid(),
                                   depth + 1),
                      alpha)
        },
        SurfaceKind::Reflective {scattering, color, alpha}  => {
            mix_color(&color,
                      &project_ray(&reflect_ray(point,
                                                &obj.normal(point),
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
    let red = Vec3(1., 0.5, 0.5);

    let background = blue;

    let intersections: Vec<(f64, Point, &Box<dyn CollisionObject>)> = s.objs.iter().filter_map(
        |s| {
            if ignore_objid == s.getid() {
                None
            } else {
                match s.intersects(&r) {
                    Some(p) => Some((p.0, p.1 , s)),
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
        let (dist, point, obj) = intersections.iter().fold(&intersections[0],
            |acc, x| {
                let (dist, _, _) = x;
                let (acc_dist, _, _) = acc;
                if dist < acc_dist { x } else { acc }
            });
            color_ray(r, s, obj, point, depth)
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
                for i in 0..self.samples_per_pixel {
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

    let v = create_viewport((width, height), (1., 1. * aspect));

    let mut objs: Vec<Box<dyn CollisionObject>> = Vec::new();
    objs.push(Box::new(Sphere {
        // orig: Vec3(-15.5 + (15.*15. - 1.*1. as f64).sqrt(),
        //            0.,
        //            6. - 1.),
        orig: Vec3(-0.2,
                   0.,
                   6.),
        r: 0.5,
        surface: SurfaceKind::Matte {
            color: Vec3(0.3, 0.3, 0.3),
            alpha: 0.5
        },
        id: 1
    }));
    objs.push(Box::new(Sphere {
        orig: Vec3(-15.5, 0., 6.),
        r: 15.,
        surface: SurfaceKind::Matte {
            color: Vec3(0.3, 0.3, 0.3),
            alpha: 0.5
        },
        id: 2
    }));
    objs.push(Box::new(Sphere {
        orig: Vec3(0.5, 0.0, 5.8),
        r: 0.2,
        // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
        // surface: SurfaceKind::Reflective(0.1),
        surface: SurfaceKind::Reflective {
            scattering: 0.1,
            color: make_color((51, 255, 165)),
            alpha: 0.7
        },
        id: 3
    }));
    objs.push(Box::new(Sphere {
        orig: Vec3(0., -1.2, 6.2),
        r: 0.5,
        // surface: SurfaceKind::Solid(Vec3(1., 0., 0.)),
        surface: SurfaceKind::Reflective {
            scattering: 0.3,
            color: make_color((255, 87, 51)),
            alpha: 0.7
        },
        id: 4
    }));

    let s = Scene {
        aspect: aspect,
        objs: &objs
    };

    v.walk_rays(&s, &mut data);

    write_png(file, (width, height), &data)
}
