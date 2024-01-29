
use core::panic;
use std::time;
use std::fs;
use std::io;
use std::io::Result;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{channel, Sender};
use std::collections::{HashMap, VecDeque};
use syscalls::{Sysno, syscall};
use core::cmp;

use crate::progress;
use crate::bitset::BitSet;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Vec3(pub f64, pub f64, pub f64);
pub type Point = Vec3;
pub type Color = Vec3;

impl Vec3 {
    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 + other.0,
             self.1 + other.1,
             self.2 + other.2,
        )
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3(self.0 - other.0,
             self.1 - other.1,
             self.2 - other.2)
    }

    pub fn mult(&self, a: f64) -> Vec3 {
        Vec3(self.0 * a,
             self.1 * a, 
             self.2 * a
        )
    }

    pub fn len2(&self) -> f64 {
        self.0*self.0 + self.1*self.1 + self.2*self.2
    }

    pub fn len(&self) -> f64 {
        self.len2().sqrt()
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.0 * other.0 +
        self.1 * other.1 +
        self.2 * other.2
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3(self.1 * other.2 - self.2 * other.1,
             self.2 * other.0 - self.0 * other.2,
             self.0 * other.1 - self.1 * other.0
        )
    }

    pub fn unit(&self) -> Vec3 {
        let res = self.mult(1./self.len());
        res
    }

    pub fn orthogonal(&self) -> Vec3 {
        if self.0.abs() > 0.1 {
            Vec3(-1.*(self.1 + self.2) / self.0, 1., 1.).unit()
        } else if self.1.abs() > 0.1 {
            Vec3(1., -1.*(self.0 + self.2) / self.1, 1.).unit()
        } else if self.2.abs() > 0.1 {
            Vec3(1., 1., -1.*(self.0 + self.1) / self.2,).unit()
        } else {
            self.unit().orthogonal()
        }
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let n = self.unit();
        let bx = n.orthogonal();
        let by = n.cross(&bx);
        (bx, by, n)
    }

    pub fn change_basis(&self, b: (Vec3, Vec3, Vec3)) -> Vec3 {
        Vec3(Vec3(b.0.0, b.0.1, b.0.2).dot(self),
             Vec3(b.1.0, b.1.1, b.1.2).dot(self),
             Vec3(b.2.0, b.2.1, b.2.2).dot(self)
        )
    }
}

pub fn make_color(color: (u8, u8, u8)) -> Color {
    Vec3((color.0 as f64) / 255.,
         (color.1 as f64) / 255.,
         (color.2 as f64) / 255.)
}

pub fn random_color() -> Color {
    Vec3(rand::random::<f64>(),
         rand::random::<f64>(),
         rand::random::<f64>())
}

fn random_vec() -> Vec3 {
    Vec3(rand::random::<f64>() - 0.5,
         rand::random::<f64>() - 0.5,
         rand::random::<f64>() - 0.5).unit()
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Ray {
    pub orig: Point,
    pub dir: Vec3,
    pub inv_dir: Vec3
}

pub fn make_ray(orig: &Point, dir: &Vec3) -> Ray {
    Ray {
        orig: *orig,
        dir: *dir,
        inv_dir: Vec3(1./dir.0,
                      1./dir.1,
                      1./dir.2)
    }
}

fn ray_intersect_helper(a: &Point, v: &Vec3, b: &Point, u:&Vec3) -> Option<(f64, f64)> {
    let det = u.0 * v.1 - u.1 * v.0;
    if det.abs() < 0.0001 {
        return None;
    }

    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    Some((
        (dy * u.0 - dx * u.1) / det,
        (dy * v.0 - dx * v.1) / det
    ))
}

impl Ray {
    fn at(&self, t: f64) -> Point {
        Vec3(self.orig.0 + self.dir.0*t,
             self.orig.1 + self.dir.1*t,
             self.orig.2 + self.dir.2*t
        )
    }

    fn intersect(&self, r: &Ray) -> Option<Point> {
        let xy_solution = ray_intersect_helper(&self.orig, &self.dir, &r.orig, &r.dir);
        let (t1, t2) = if xy_solution.is_some() {
            xy_solution.unwrap()
        } else {
            let xz_solution = ray_intersect_helper(
                &Vec3(self.orig.0, self.orig.2, self.orig.1),
                &Vec3(self.dir.0, self.dir.2, self.dir.1),
                &Vec3(r.orig.0, r.orig.2, r.orig.1),
                &Vec3(r.dir.0, r.dir.2, r.dir.1));
            if xz_solution.is_some() {
                xz_solution.unwrap()
            } else {
                let yz_solution = ray_intersect_helper(
                &Vec3(self.orig.1, self.orig.2, self.orig.0),
                &Vec3(self.dir.1, self.dir.2, self.dir.0),
                &Vec3(r.orig.1, r.orig.2, r.orig.0),
                &Vec3(r.dir.1, r.dir.2, r.dir.0));

                if yz_solution.is_some() {
                    yz_solution.unwrap()
                } else {
                    return None;
                    // (-1., -1.)
                }
            }
        };

        let p1 = self.at(t1);
        let p2 = r.at(t2);

        let p_diff = p2.sub(&p1);
        if p_diff.len2() < 0.01{
            Some(p1)
        } else {
            None
        }
    }

    fn nearest_point(&self, o: &Point) -> Point {

        let ao = o.sub(&self.orig);
        let ap = self.dir.mult(ao.dot(&self.dir)/self.dir.len2());

        ap.add(&self.orig)
    }
}

fn reflect_ray(orig: &Point, norm: &Vec3, dir: &Vec3, fuzz: f64) -> Ray {

    let ddot = dir.dot(norm).abs();
    let dir_p = norm.mult(ddot);
    let dir_o  = dir.add(&dir_p);

    let reflect = dir_p.add(&dir_o);
    let rand_vec = random_vec().mult(fuzz);

    make_ray(orig, &reflect.add(&rand_vec).unit())
}

fn lambertian_ray(orig: &Point, norm: &Vec3) -> Ray {

    let rand_vec = random_vec();

    make_ray(orig, &norm.add(&rand_vec))
}

fn mix_color(c1: &Color, c2: &Color, a: f64) -> Color {
    c1.mult(1. - a).add(&c2.mult(a))
}

#[derive(Copy, Clone, Debug)]
pub enum SurfaceKind {
    Solid { color: Color },
    Matte { color: Color, alpha: f64},
    Reflective { scattering: f64, color: Color, alpha: f64},
}

#[derive(Debug, PartialEq)]
pub enum CollisionFace {
    Front,
    Back,
    Side,
    Face(usize),
    EdgeFront,
    EdgeBack
}

pub trait Collidable {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)>;
    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3;
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind;
    fn getid(&self) -> usize;
}

#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    pub orig: Point,
    pub r: f64,
    pub surface: SurfaceKind,
    pub id: usize
}

impl Collidable for Sphere {
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

    fn getid(&self) -> usize {
        self.id
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Disk {
    pub orig: Point,
    pub norm: Vec3,
    pub r: f64,
    pub depth: f64,
    pub surface: SurfaceKind,
    pub side_surface: SurfaceKind,
    pub id: usize
}

impl Collidable for Disk {
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

            let r_n_temp = make_ray(&r.orig.sub(&self.orig).change_basis(n_basis),
                                    &r.dir.change_basis(n_basis));
            let r_n = make_ray(&Vec3(r_n_temp.orig.0, r_n_temp.orig.1, 0.),
                               &Vec3(r_n_temp.dir.0, r_n_temp.dir.1, 0.));

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
    fn getid(&self) -> usize {
        self.id
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Triangle {
    pub incenter: Point,
    pub norm: Vec3,
    pub sides: [Vec3; 3],
    pub side_lens: [f64; 3],
    // sides: [(Vec3, f64); 3],
    pub corners: [Vec3; 3],
    pub surface: SurfaceKind,
    pub edge_thickness: f64,
    pub id: usize
}

pub fn make_polygon(points: &[Vec3; 3], surface: &SurfaceKind, edge_thickness: f64, id: usize) -> Triangle {

    let a = points[0];
    let b = points[points.len()/3];
    let c = points[(points.len()*2)/3];

    // println!("A {:?}", a);
    // println!("B {:?}", b);
    // println!("C {:?}", c);

    let ab = b.sub(&a);
    let ac = c.sub(&a);
    let bc = c.sub(&b);

    let bac_bisect = ac.add(&ab);
    let abc_bisect = bc.add(&ab.mult(-1.));

    let bac_bi_ray = make_ray(&a, &bac_bisect);

    let abc_bi_ray = make_ray(&b, &abc_bisect);

    let incenter = bac_bi_ray.intersect(&abc_bi_ray).unwrap();

    let mut sides = [Vec3(0.,0.,0.); 3];
    let mut side_lens = [0.; 3];
    for idx in 0..points.len() {
        let vedge = points[(idx+1)%points.len()].sub(&points[idx]);
        let po = incenter.sub(&points[idx]);
        let pc = vedge.mult(vedge.dot(&po)/vedge.len2());
        let oc = pc.sub(&po);
        sides[idx] = oc.unit();
        side_lens[idx] = oc.len();
        // sides.push((oc.unit(), oc.len()));
    }

    let norm = sides[0].cross(&sides[1]).unit();

    // let ai = incenter.sub(&a);
    // let bi = incenter.sub(&b);
    // let ci = incenter.sub(&c);

    // let acs = ab.mult(ab.dot(&ai)/ab.len2());
    // let abs = ac.mult(ac.dot(&ai)/ac.len2());
    // let bcs = bc.mult(bc.dot(&bi)/bc.len2());

    // let ias = bcs.sub(&bi);
    // let ibs = abs.sub(&ai);
    // let ics = acs.sub(&ai);

    // let bounding_len2 = ai.len2().max(bi.len2()).max(ci.len2());

    // let norm = ab.cross(&ac).unit();
    // println!("\nNorms:");
    // println!(" {:?}", norm);
    // println!(" {:?}", corner_norms.0);
    // println!(" {:?}", corner_norms.1);
    // println!(" {:?}", corner_norms.2);

    Triangle {
        incenter: incenter,
        norm: norm,
        // bounding_r2: bounding_len2,
        sides: sides,
        side_lens: side_lens,
        corners: points.clone(),
        surface: *surface,
        edge_thickness: edge_thickness,
        id: id
    }
}

impl Collidable for Triangle {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {

        let t = self.norm.dot(&self.incenter.sub(&r.orig)) / self.norm.dot(&r.dir);
        if t < 0. {
            return None;
        }
        let p = r.at(t);

        let ip = p.sub(&self.incenter);

        // if ip.len2() > self.bounding_r2 {
        //     return None;
        // }

        let mut hit_edge = false;
        for (side, side_len) in self.sides.iter().zip(self.side_lens) {
            let dist =  ip.dot(&side);
            // println!("{:?} {} {:?} {:?}", dist, side_len, side, ip);
            if dist > side_len {
                return None;
            } else if dist > (side_len * self.edge_thickness) {
                hit_edge = true;
            }
        }

        // if ip.dot(&self.side_a) > self.side_a_d ||
        //    ip.dot(&self.side_b) > self.side_b_d ||
        //    ip.dot(&self.side_c) > self.side_c_d {
        //     return None;
        // }

        let side = if hit_edge {
            if r.dir.dot(&self.norm) > 0. {
                CollisionFace::EdgeBack
            } else {
                CollisionFace::EdgeFront
            }
        } else {
            if r.dir.dot(&self.norm) > 0. {
                CollisionFace::Back
            } else {
                CollisionFace::Front
            }
        };

        // let side = if ip.dot(&self.side_a) > self.side_a_d*self.edge ||
        //               ip.dot(&self.side_b) > self.side_b_d*self.edge ||
        //               ip.dot(&self.side_c) > self.side_c_d*self.edge {
        //     CollisionFace::Edge
        // } else {
        //     if r.dir.dot(&self.norm) > 0. {
        //         CollisionFace::Back
        //     } else {
        //         CollisionFace::Front
        //     }
        // };

        Some((t, p, side))
    }

    fn normal(&self, _p: &Point, f: &CollisionFace) -> Vec3 {
        match f {
            CollisionFace::Front => self.norm,
            CollisionFace::Back => self.norm.mult(-1.),
            CollisionFace::EdgeFront => self.norm,
            CollisionFace::EdgeBack => self.norm.mult(-1.),
            _ => panic!("Invalid face for triangle")
        }
    }
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match f {
            CollisionFace::EdgeFront => {
                SurfaceKind::Solid { color: make_color((0, 0, 0)) }
            },
            CollisionFace::EdgeBack => {
                SurfaceKind::Solid { color: make_color((0, 0, 0)) }
            },
            _ => self.surface
        }
    }
    fn getid(&self) -> usize {
        self.id
    }
}


#[derive(Clone, Debug)]
pub enum CollisionObject {
    Sphere(Sphere),
    Triangle(Triangle),
    Disk(Disk),
}

impl Collidable for CollisionObject {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {
        match self {
            CollisionObject::Sphere(s) => s.intersects(r),
            CollisionObject::Triangle(t) => t.intersects(r),
            CollisionObject::Disk(d) => d.intersects(r),
        }
    }

    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3 {
        match self {
            CollisionObject::Sphere(s) => s.normal(p, f),
            CollisionObject::Triangle(t) => t.normal(p, f),
            CollisionObject::Disk(d) => d.normal(p, f),
        }
    }

    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match self {
            CollisionObject::Sphere(s) => s.getsurface(f),
            CollisionObject::Triangle(t) => t.getsurface(f),
            CollisionObject::Disk(d) => d.getsurface(f),
        }
    }

    fn getid(&self) -> usize {
        match self {
            CollisionObject::Sphere(s) => s.getid(),
            CollisionObject::Triangle(t) => t.getid(),
            CollisionObject::Disk(d) => d.getid(),
        }
    }
}

pub struct LightSource {
    pub orig: Point,
    pub len2: f64
}

impl LightSource {
    fn _get_shadow_ray(&self, p: &Point, norm: &Vec3) -> Ray {
        
        let adj_orig = Vec3(self.orig.0 + rand::random::<f64>() * self.len2,
                            self.orig.1 + rand::random::<f64>() * self.len2,
                            self.orig.2 + rand::random::<f64>() * self.len2);
        let dir = adj_orig.sub(p).unit();
        let smudge = norm.mult(0.005 * (rand::random::<f64>() + 1.));
        make_ray(&p.add(&smudge), &dir)
    }
}

#[repr(C)]
pub struct BoundingBox {
    pub orig: Point,
    pub len2: f64,
    pub objs: Vec<usize>,
    pub child_boxes: Vec<BoundingBox>,
    pub depth: usize
}

impl Clone for BoundingBox {
    fn clone(&self) -> Self {
        BoundingBox {
            orig: self.orig,
            len2: self.len2,
            objs: self.objs.clone(),
            child_boxes: self.child_boxes.clone(),
            depth: self.depth
        }
    }
}

fn box_contains_point(orig: &Point, len2: f64, p: &Point) -> bool {

    let op = p.sub(orig);

    op.0.abs() < len2 &&
    op.1.abs() < len2 &&
    op.2.abs() < len2
}


pub fn box_contains_polygon(orig: &Point, len2: f64, t: &Triangle) -> bool {

    if box_contains_point(orig, len2, &t.incenter) {
        return true;
    }

    for corner in &t.corners {
        if box_contains_point(orig, len2, &corner) {
            return true;
        }
    }
    // if box_contains_point(orig, len2, &t.corners.0) ||
    //    box_contains_point(orig, len2, &t.corners.1) ||
    //    box_contains_point(orig, len2, &t.corners.2) {
    //     return true;
    // }

    let ab_ray = make_ray(&t.corners[0], &t.corners[1].sub(&t.corners[0]));
    let ab_p = ab_ray.nearest_point(orig);
    let ap1 = ab_p.sub(&ab_ray.orig);
    let ab_v = ap1.dot(&ab_ray.dir);
    if box_contains_point(orig, len2, &ab_p) &&
        ab_v >= 0. && ab_v < ab_ray.dir.len2() {
        return true;
    }

    let ac_ray = make_ray(&t.corners[0], &t.corners[2].sub(&t.corners[0]));
    let ac_p = ac_ray.nearest_point(orig);
    let ap2 = ac_p.sub(&ac_ray.orig);
    let ac_v = ap2.dot(&ac_ray.dir);
    if box_contains_point(orig, len2, &ac_p) &&
        ac_v >= 0. && ac_v < ac_ray.dir.len2() {
        return true;
    }

    let bc_ray = make_ray(&t.corners[1], &t.corners[2].sub(&t.corners[1]));
    let bc_p = bc_ray.nearest_point(orig);
    let bp = bc_p.sub(&bc_ray.orig);
    let bc_v = bp.dot(&bc_ray.dir);
    if box_contains_point(orig, len2, &bc_p) &&
        bc_v > 0. && bc_v < bc_ray.dir.len2() {
        return true;
    }

    // Check a ray for each edge of the cube
    let mut points = [Vec3(0., 0., 0.); 8];
    for i in 0..8 {
        let xoff = if (i & 1) == 0 { len2 } else {-1.*len2};
        let yoff = if (i & 2) == 0 { len2 } else {-1.*len2};
        let zoff = if (i & 4) == 0 { len2 } else {-1.*len2};
        points[i] = Vec3(orig.0 + xoff, orig.1 + yoff, orig.2 + zoff);
    }

    let edges: [(usize, usize); 12] = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ];

    for edge in edges {
        let edge_ray = make_ray(&points[edge.0], &points[edge.1].sub(&points[edge.0]));
        match t.intersects(&edge_ray) {
            Some((tt, _, __)) => {
                if tt >= 0. && tt <= 1. {
                    return true;
                }
            }
            None => {}
        };
    }


    false
}

pub fn build_empty_box() -> BoundingBox {
    BoundingBox {
        orig: Vec3(0., 0., 0.),
        len2: 1.,
        objs: Vec::new(),
        child_boxes: Vec::new(),
        depth: 0
    }
}

pub fn build_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f64, maxdepth: usize, minobjs: usize) -> BoundingBox {
    let mut refvec: Vec<usize> = Vec::with_capacity(tris.len());
    refvec.extend(0..tris.len());
    build_bounding_box_helper(tris, &refvec, orig, len2, 0, maxdepth, minobjs)
}

fn build_bounding_box_helper(tris: &Vec<Triangle>, objs: &Vec<usize>, orig: &Point, len2: f64, depth: usize, maxdepth: usize, minobjs: usize) -> BoundingBox {

    let mut subobjs: Vec<usize> = Vec::with_capacity(objs.len());
    for idx in objs {
        if box_contains_polygon(orig, len2, &tris[*idx]) {
            subobjs.push(*idx);
        }
    }

    if subobjs.len() == 0 {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: Vec::new(),
            child_boxes: Vec::new(),
            depth: depth
        };
    } else if subobjs.len() < minobjs || depth >= maxdepth {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: subobjs,
            child_boxes: Vec::new(),
            depth: depth
        };
    }

    let mut subboxes: Vec<BoundingBox> = Vec::with_capacity(8);
    let newlen2 = len2 / 2.;
    for i in 0..8 {
        let xoff = if (i & 1) == 0 { -1.*newlen2 } else { newlen2 } ;
        let yoff = if (i & 2) == 0 { -1.*newlen2 } else { newlen2 } ;
        let zoff = if (i & 4) == 0 { -1.*newlen2 } else { newlen2 } ;
        let bbox = build_bounding_box_helper(tris,
                                             &subobjs,
                                             &Vec3(orig.0 + xoff,
                                                   orig.1 + yoff,
                                                   orig.2 + zoff),
                                             newlen2,
                                             depth + 1,
                                             maxdepth,
                                             minobjs);
        subboxes.push(bbox);
    }

    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: Vec::new(),
        child_boxes: subboxes,
        depth: depth
    }
}

pub fn build_trivial_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f64) -> BoundingBox {

    let mut refvec: Vec<usize> = Vec::with_capacity(tris.len());
    refvec.extend::<Vec<usize>>((0..tris.len()).collect());
    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: refvec,
        child_boxes: Vec::new(),
        depth: 0
    }
}

impl BoundingBox {
    
    fn collides_face(&self, r: &Ray, face: usize) -> bool {
        
        let norm = match face {
            0 => Vec3(1., 0., 0.),
            1 => Vec3(-1., 0., 0.),
            2 => Vec3(0., 1., 0.),
            3 => Vec3(0., -1., 0.),
            4 => Vec3(0., 0., 1.),
            5 => Vec3(0., 0., -1.),
            _ => panic!("Invalid face {}", face)
        };

        let ortho_vecs = match face {
            0 => (Vec3(0., 1., 0.), Vec3(0., 0., 1.)),
            1 => (Vec3(0., 1., 0.), Vec3(0., 0., 1.)),
            2 => (Vec3(1., 0., 0.), Vec3(0., 0., 1.)),
            3 => (Vec3(1., 0., 0.), Vec3(0., 0., 1.)),
            4 => (Vec3(1., 0., 0.), Vec3(0., 1., 0.)),
            5 => (Vec3(1., 0., 0.), Vec3(0., 1., 0.)),
            _ => panic!("Invalid face {}", face)
        };

        // t = -1 * (norm * (orig - Rorig)) / (norm * Rdir)
        let surf = self.orig.add(&norm.mult(self.len2));
        let t = norm.dot(&surf.sub(&r.orig)) / norm.dot(&r.dir);
        if t < 0. {
            return false;
        }

        let p = r.at(t);
        let op = p.sub(&self.orig);
        
        op.dot(&ortho_vecs.0).abs() < self.len2 &&
        op.dot(&ortho_vecs.1).abs() < self.len2
    }

    pub fn collides(&self, r: &Ray) -> bool {
        // let p = r.nearest_point(&self.orig);
        // let op = p.sub(&self.orig);

        // let hit = (op.0.abs() < self.len2 &&
        //            op.1.abs() < self.len2 &&
        //            op.2.abs() < self.len2) ||
        //            self.collides_face(r, 0) ||
        //            self.collides_face(r, 1) ||
        //            self.collides_face(r, 2) ||
        //            self.collides_face(r, 3) ||
        //            self.collides_face(r, 4) ||
        //            self.collides_face(r, 5);

        // hit
    
        let mut tmin = f64::MIN;
        let mut tmax = f64::MAX;
        if r.dir.0 != 0. {
            let t1 = (self.orig.0 - self.len2 - r.orig.0) * r.inv_dir.0;
            let t2 = (self.orig.0 + self.len2 - r.orig.0) * r.inv_dir.0;

            tmin = f64::max(tmin, f64::min(t1, t2));
            tmax = f64::min(tmax, f64::max(t1, t2));
        }

        if r.dir.1 != 0. {
            let t1 = (self.orig.1 - self.len2 - r.orig.1) * r.inv_dir.1;
            let t2 = (self.orig.1 + self.len2 - r.orig.1) * r.inv_dir.1;

            tmin = f64::max(tmin, f64::min(t1, t2));
            tmax = f64::min(tmax, f64::max(t1, t2));
        }

        if r.dir.2 != 0. {
            let t1 = (self.orig.2 - self.len2 - r.orig.2) * r.inv_dir.2;
            let t2 = (self.orig.2 + self.len2 - r.orig.2) * r.inv_dir.2;

            tmin = f64::max(tmin, f64::min(t1, t2));
            tmax = f64::min(tmax, f64::max(t1, t2));
        }

        tmin < tmax
    }

    pub fn get_all_objects_for_ray(&self, tris: &Vec<Triangle>, r: &Ray, path: &mut Vec<BoundingBox>) -> BitSet {

        let mut objmap = BitSet::new(tris.len());

        self.get_all_objects_for_ray_helper(tris, r, &mut objmap, path);

        objmap
    }

    fn get_all_objects_for_ray_helper(&self, tris: &Vec<Triangle>, r: &Ray, objmap: &mut BitSet, path: &mut Vec<BoundingBox>) {
        if self.objs.len() == 0&&
           self.child_boxes.len() == 0{
            return;
        }
        if self.collides(r) {
            if self.objs.len() > 0 {
                objmap.extend(self.objs.iter());
            }

            if self.child_boxes.len() > 0 {
                for cbox in &self.child_boxes {
                    cbox.get_all_objects_for_ray_helper(tris, r, objmap, path);
                }
            }
        }
    }

    pub fn print_tree(&self) {
        self.print();
        if self.child_boxes.len() > 0 {
            for b in &self.child_boxes {
                b.print_tree();
            }
        }
    }

    pub fn print(&self) {
        println!("BS: {} {:?} {} {} {}",
                 self.depth,
                 self.orig, self.len2,
                 self.objs.len(),
                 self.child_boxes.len()
                 );
    }

    // pub fn find_obj(&self, tris: &Vec<Triangle>, id: usize) -> Vec<&BoundingBox> {
    //     let mut found_list: Vec<&BoundingBox> = Vec::new();
    //     self.find_obj_helper(tris, id, &mut found_list);
    //     found_list
    // }

    // pub fn find_obj_helper(&self, tris: &Vec<Triangle>, id: usize, found_list: &mut Vec<&BoundingBox>) {
    //     if self.objs.len() > 0 {
    //         for obj in &self.objs {
    //             if tris[*obj].getid() == id {
    //                 found_list.push(&self);
    //             }
    //         }
    //     }

    //     if self.child_boxes.len() > 0 {
    //         for b in &self.child_boxes {
    //             b.find_obj_helper(tris, id, found_list);
    //         }
    //     }
    // }

}

pub trait RayCaster: Send + Sync {
    fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
                   depth: usize, runtimes: &mut HashMap<String, time::Duration>) -> Color;
    fn color_ray(&self, r: &Ray, s: &Scene, objidx: usize,
                 point: &Point, face: &CollisionFace, depth: usize,
                 runtimes: &mut HashMap<String, time::Duration>) -> Color;
}

#[derive(Debug)]
#[repr(C)]
struct LinuxTimespec {
    tv_sec: i64,
    tv_nsec: i64
}

pub fn get_thread_time() -> i64 {

    let mut t = LinuxTimespec {
        tv_sec: 0,
        tv_nsec: 0
    };
    match unsafe { syscall!(Sysno::clock_gettime, 3 /*CLOCK_THREAD_CPUTIME_ID*/, &mut t as *mut LinuxTimespec) } {
        Ok(_r) => {
        } 
        Err(err) => {
            panic!("clock_gettime() failed: {}", err)
        }
    };

    t.tv_sec * (1000*1000*1000) + t.tv_nsec
}

#[derive(Copy,Clone)]
pub struct DefaultRayCaster {
}

unsafe impl Send for DefaultRayCaster {}
unsafe impl Sync for DefaultRayCaster {}

impl RayCaster for DefaultRayCaster {
    fn color_ray(&self, r: &Ray, s: &Scene, objidx: usize,
                 point: &Point, face: &CollisionFace, depth: usize, runtimes: &mut HashMap<String, time::Duration>) -> Color {

        let shadowed = false;
        // let shadowed = if s.lights.is_some() {
        //     let norm = obj.normal(point, face);
        //     let light_ray = s.lights.as_ref().unwrap().get_shadow_ray(point, &norm);

        //     let mut path: Vec<BoundingBox> = Vec::new();
        //     let mut objs = s.boxes.get_all_objects_for_ray(&light_ray, &mut path);
        //     // objs.extend(s.otherobjs.iter().map(|o| (o.getid(), o)));

        //     let mut found = false;
        //     for (id, search_obj) in objs {
        //         if id != obj.getid() {
        //             if search_obj.intersects(&light_ray).is_some() {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //     }

        //     found
        // } else {
        //     false
        // };

        let black = make_color((0,0,0));

        match s.tris[objidx].getsurface(face) {
            SurfaceKind::Solid {color} => {
                if !shadowed {color} else {black}
            },
            SurfaceKind::Matte {color, alpha} => {
                mix_color(if !shadowed {&color} else {&black},
                        &self.project_ray(&lambertian_ray(point,
                                                    &s.tris[objidx].normal(point, face)),
                                    s,
                                    s.tris[objidx].getid(),
                                    depth + 1,
                                    runtimes),
                        alpha)
                
            },
            SurfaceKind::Reflective {scattering, color, alpha}  => {
                mix_color(if !shadowed {&color} else {&black},
                        &self.project_ray(&reflect_ray(point,
                                                    &s.tris[objidx].normal(point, face),
                                                    &r.dir,
                                                    scattering),
                                    s,
                                    s.tris[objidx].getid(),
                                    depth + 1,
                                    runtimes),
                        alpha)
            }
        }
    }

    fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
                   depth: usize, runtimes: &mut HashMap<String, time::Duration>) -> Color {

        if depth == 0 {
            return Vec3(0., 0., 0.);
        }

        let t1 = get_thread_time();

        let blue = Vec3(0.5, 0.7, 1.);

        let mut path: Vec<BoundingBox> = Vec::new();
        let objs = s.boxes.get_all_objects_for_ray(&s.tris, r, &mut path);
        // objs.extend(s.otherobjs.iter().map(|o| (o.getid(), o)));

        let t2 = get_thread_time();

        let intersections: Vec<(f64, Point, CollisionFace, usize)> = objs.iter().filter_map(
            |idx| {
                if ignore_objid == idx as usize {
                    None
                } else {
                    match s.tris[idx as usize].intersects(&r) {
                        Some(p) => Some((p.0, p.1, p.2, idx as usize)),
                        None    => None
                    }
                }
            }).collect();
        
        let t3 = get_thread_time();

        *runtimes.entry("BoundingBox".to_string()).or_default() += time::Duration::from_nanos((t2-t1) as u64);
        *runtimes.entry("Intersections".to_string()).or_default() += time::Duration::from_nanos((t3-t2) as u64);

        if intersections.len() == 0 {
            blue
        } else {
            let (_dist, point, face, objidx) = intersections.iter().fold(&intersections[0],
                |acc, x| {
                    let (dist, _, _, _) = x;
                    let (acc_dist, _, _, _) = acc;
                    if dist < acc_dist { x } else { acc }
                });
                self.color_ray(r, s, *objidx, point, &face, depth, runtimes)

        }
    }

}

#[repr(C)]
pub struct Scene {
    pub tris: Vec<Triangle>,
    pub boxes: BoundingBox,
    // pub otherobjs: Vec<CollisionObject>,
    // pub lights: Option<LightSource>
}

#[derive(Clone, Copy,Debug)]
#[repr(C)]
pub struct Viewport {
    width: usize,
    height: usize,

    orig: Point,
    cam: Point,
    
    vu: Vec3,
    vv: Vec3,

    maxdepth: usize,
    samples_per_pixel: usize
}

pub fn create_transform(dir_in: &Vec3, d_roll: f64) -> (Vec3, Vec3, Vec3) {

    let dir = dir_in.unit();

    let roll = -1.*(-1. * dir.1).atan2(dir.2);
    let pitch = -1.*dir.0.asin();
    let yaw = -1.*d_roll;

    (
        Vec3(yaw.cos()*pitch.cos(),
             yaw.sin()*pitch.cos(),
             -1.*pitch.sin()),

        Vec3(yaw.cos()*pitch.sin()*roll.sin() - yaw.sin()*roll.cos(),
             yaw.sin()*pitch.sin()*roll.sin() + yaw.cos()*roll.cos(),
             pitch.cos()*roll.sin()),

        Vec3(yaw.cos()*pitch.sin()*roll.cos() + yaw.sin()*roll.sin(),
             yaw.sin()*pitch.sin()*roll.cos() - yaw.cos()*roll.sin(),
             pitch.cos()*roll.cos())
    )
}

pub fn create_viewport(px: (u32, u32), size: (f64, f64), pos: &Point, dir: &Vec3, fov: f64, c_roll: f64, maxdepth: usize, samples: usize) -> Viewport {

    let dist = size.0 / (2. * (fov.to_radians() / 2.).tan());

    let rot_basis = create_transform(dir, c_roll);

    let orig = pos.add(&Vec3(1.*size.1/2., -1.*size.0/2., 0.));
    
    let cam_r = Vec3(0., 0., dist).change_basis(rot_basis);
    let cam = pos.sub(&cam_r);

    let vu = Vec3(0., size.0, 0.);
    let vu_r = vu.change_basis(rot_basis);

    let vv = Vec3(-1. * size.1, 0., 0.);
    let vv_r = vv.change_basis(rot_basis);

    Viewport {
        width: px.0 as usize,
        height: px.1 as usize,
        orig: orig,
        cam: cam,
        vu: vu_r,
        vv: vv_r,
        maxdepth: maxdepth,
        samples_per_pixel: samples
    }
}

impl Viewport {

    fn pixel_ray(&self, px: (usize, usize)) -> Ray {

        let px_x = px.0 as f64;
        let px_y = px.1 as f64;

        let vu_delta = self.vu.mult(1. / self.width as f64);
        let vv_delta = self.vv.mult(1. / self.height as f64);

        // let u_off: f64 = 0.5;
        // let v_off: f64 = 0.5;

        let u_off: f64 = rand::random();
        let v_off: f64 = rand::random();

        let vu_frac = vu_delta.mult(px_y + u_off);
        let vv_frac = vv_delta.mult(px_x + v_off);

        let px_u = self.orig.add(&vu_frac).add(&vv_frac);

        make_ray(&px_u, &px_u.sub(&self.cam).unit())
    }

    fn walk_ray_set(&self, s: &Scene, rows: Arc<Mutex<VecDeque<(&mut [Color], usize)>>>,
                    t_num: usize, progress_tx: Sender<(usize, usize, usize, usize, HashMap<String, time::Duration>)>,
                    caster: &dyn RayCaster) {

        let mut rays_count = 0;
        let mut pixels_processed = 0;
        let mut runtimes: HashMap<String, time::Duration> = HashMap::new();
        'threadloop: loop {
            let (data, row) = match rows.lock().unwrap().pop_front() {
                                Some(x) => {
                                    (x.0, x.1)
                                }
                                None => break 'threadloop
                                };
            
            
            let _ = progress_tx.send((t_num, row, 0, 0, runtimes.clone()));
            runtimes.clear();
            for y in 0..self.width {
                let mut acc = Vec3(0., 0., 0.);
                for _i in 0..self.samples_per_pixel {
                    let ray_color = caster.project_ray(&self.pixel_ray((row,y)), s, 0, self.maxdepth, &mut runtimes);
                    acc = acc.add(&ray_color);

                    rays_count += 1;
                }
                data[y as usize] = acc.mult(1./(self.samples_per_pixel as f64));
                pixels_processed += 1;
                if rays_count > 10000 {
                    let _ = progress_tx.send((t_num, row, pixels_processed, rays_count, runtimes.clone()));
                    rays_count = 0;
                    pixels_processed = 0;
                    runtimes.clear();
                }
            }
            let _ = progress_tx.send((t_num, row, pixels_processed, rays_count, runtimes.clone()));
            rays_count = 0;
            pixels_processed = 0;
            runtimes.clear();
        }
    }

    pub fn walk_rays(&self, s: &Scene, data: & mut[Color], threads: usize, caster: &dyn RayCaster) -> progress::ProgressCtx {

        let (progress_tx, progress_rx) = channel();

        let mut progress_io = progress::create_ctx(threads, self.width, self.height, true);


        thread::scope(|sc| {

            let data_parts: Arc<Mutex<VecDeque<(&mut [Color], usize)>>> = 
                Arc::new(Mutex::new(VecDeque::from(
                    data.chunks_mut(self.width).zip(0..self.height).collect::<VecDeque<(&mut [Color], usize)>>())));

            for t in 0..threads {
                let t_progress_tx = progress_tx.clone();
                let t_parts = Arc::clone(&data_parts);
                sc.spawn(move || {
                    self.walk_ray_set(s, t_parts, t, t_progress_tx, caster);
                });
            }

            drop(progress_tx);
            'waitloop: loop {
                match progress_rx.recv() {
                    Ok((t_num, row, y, rays_so_far, runtimes)) => {
                        progress_io.update(t_num, row, y, rays_so_far, &runtimes);
                    }
                    Err(_x) => {
                        break 'waitloop;
                    }
                };
            }
        });

        progress_io.finish();

        progress_io
    }
}


pub fn write_png(f: fs::File, img_size: (u32, u32), data: &[Color]) -> Result<()> {

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
