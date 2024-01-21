
use core::panic;
use std::fs;
use std::hash::Hash;
use std::io;
use std::io::Result;
use std::ops::Bound;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::ops::Range;
use std::sync::mpsc::{channel, Sender};
use std::collections::HashMap;
use std::time;
use std::time::Instant;

use crate::progress;
use crate::progress::ProgressCtx;

#[derive(Copy, Clone, Debug)]
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
pub struct Ray {
    pub orig: Point,
    pub dir: Vec3,
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
        let mut t1: f64 = 0.;
        let mut t2: f64 = 0.;

        let xy_solution = ray_intersect_helper(&self.orig, &self.dir, &r.orig, &r.dir);
        if xy_solution.is_some() {
            (t1, t2) = xy_solution.unwrap();
        } else {
            let xz_solution = ray_intersect_helper(
                &Vec3(self.orig.0, self.orig.2, self.orig.1),
                &Vec3(self.dir.0, self.dir.2, self.dir.1),
                &Vec3(r.orig.0, r.orig.2, r.orig.1),
                &Vec3(r.dir.0, r.dir.2, r.dir.1));
            if xz_solution.is_some() {
                (t1, t2) = xz_solution.unwrap();
            } else {
                let yz_solution = ray_intersect_helper(
                &Vec3(self.orig.1, self.orig.2, self.orig.0),
                &Vec3(self.dir.1, self.dir.2, self.dir.0),
                &Vec3(r.orig.1, r.orig.2, r.orig.0),
                &Vec3(r.dir.1, r.dir.2, r.dir.0));

                if yz_solution.is_some() {
                    (t1, t2) = yz_solution.unwrap();
                } else {
                    return None;
                }
            }
        }

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
    Edge
}

pub trait Collidable {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)>;
    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3;
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind;
    fn getid(&self) -> usize;
}

#[derive(Copy, Clone)]
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

#[derive(Copy, Clone)]
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

            let r_n_temp = Ray {
                orig: r.orig.sub(&self.orig).change_basis(n_basis),
                dir: r.dir.change_basis(n_basis),
            };
            let r_n = Ray {
                orig: Vec3(r_n_temp.orig.0, r_n_temp.orig.1, 0.),
                dir: Vec3(r_n_temp.dir.0, r_n_temp.dir.1, 0.),
            };

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
pub struct Triangle {
    incenter: Point,
    norm: Vec3,
    bounding_r2: f64,
    side_a: Vec3,
    side_a_d: f64,
    side_b: Vec3,
    side_b_d: f64,
    side_c: Vec3,
    side_c_d: f64,
    corners: (Point, Point, Point),
    surface: SurfaceKind,
    edge: f64,
    pub id: usize
}

pub fn make_triangle(points: (Vec3, Vec3, Vec3), surface: &SurfaceKind, edge: f64, id: usize) -> Triangle {

    let a = points.0;
    let b = points.1;
    let c = points.2;

    let ab = b.sub(&a);
    let ac = c.sub(&a);
    let bc = c.sub(&b);

    let bac_bisect = ac.add(&ab);
    let abc_bisect = bc.add(&ab.mult(-1.));

    let bac_bi_ray = Ray {
        orig: a,
        dir: bac_bisect
    };

    let abc_bi_ray = Ray {
        orig: b,
        dir: abc_bisect
    };

    let incenter = bac_bi_ray.intersect(&abc_bi_ray).unwrap();

    let ai = incenter.sub(&a);
    let bi = incenter.sub(&b);
    let ci = incenter.sub(&c);

    let acs = ab.mult(ab.dot(&ai)/ab.len2());
    let abs = ac.mult(ac.dot(&ai)/ac.len2());
    let bcs = bc.mult(bc.dot(&bi)/bc.len2());

    let ias = bcs.sub(&bi);
    let ibs = abs.sub(&ai);
    let ics = acs.sub(&ai);

    let bounding_len2 = ai.len2().max(bi.len2()).max(ci.len2());

    Triangle {
        incenter: incenter,
        norm: ab.cross(&ac).unit(),
        bounding_r2: bounding_len2,
        side_a: ias.unit(),
        side_a_d: ias.len(),
        side_b: ibs.unit(),
        side_b_d: ibs.len(),
        side_c: ics.unit(),
        side_c_d: ics.len(),
        corners: points,
        surface: *surface,
        edge: edge,
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

        if ip.len2() > self.bounding_r2 {
            return None;
        }

        if ip.dot(&self.side_a) > self.side_a_d ||
           ip.dot(&self.side_b) > self.side_b_d ||
           ip.dot(&self.side_c) > self.side_c_d {
            return None;
        }

        let side = if ip.dot(&self.side_a) > self.side_a_d*self.edge ||
                      ip.dot(&self.side_b) > self.side_b_d*self.edge ||
                      ip.dot(&self.side_c) > self.side_c_d*self.edge {
            CollisionFace::Edge
        } else {
            if r.dir.dot(&self.norm) > 0. {
                CollisionFace::Back
            } else {
                CollisionFace::Front
            }
        };

        Some((t, p, side))
    }

    fn normal(&self, _p: &Point, f: &CollisionFace) -> Vec3 {
        match f {
            CollisionFace::Front => self.norm,
            CollisionFace::Back => self.norm.mult(-1.),
            _ => panic!("Invalid face for triangle")
        }
    }
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match f {
            CollisionFace::Edge => {
                SurfaceKind::Solid { color: make_color((0, 0, 0)) }
            },
            _ => self.surface
        }
    }
    fn getid(&self) -> usize {
        self.id
    }
}


#[derive(Copy,Clone)]
pub enum CollisionObject {
    Sphere(Sphere),
    Triangle(Triangle),
    Disk(Disk),
}

impl Collidable for CollisionObject {
    fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {
        match *self {
            CollisionObject::Sphere(s) => s.intersects(r),
            CollisionObject::Triangle(t) => t.intersects(r),
            CollisionObject::Disk(d) => d.intersects(r),
        }
    }

    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3 {
        match *self {
            CollisionObject::Sphere(s) => s.normal(p, f),
            CollisionObject::Triangle(t) => t.normal(p, f),
            CollisionObject::Disk(d) => d.normal(p, f),
        }
    }

    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match *self {
            CollisionObject::Sphere(s) => s.getsurface(f),
            CollisionObject::Triangle(t) => t.getsurface(f),
            CollisionObject::Disk(d) => d.getsurface(f),
        }
    }

    fn getid(&self) -> usize {
        match *self {
            CollisionObject::Sphere(s) => s.getid(),
            CollisionObject::Triangle(t) => t.getid(),
            CollisionObject::Disk(d) => d.getid(),
        }
    }
}

pub struct BoundingBox<'a> {
    orig: Point,
    len2: f64,
    objs: Option<Vec<&'a CollisionObject>>,
    child_boxes: Option<Vec<BoundingBox<'a>>>,
    depth: usize
}

impl Clone for BoundingBox<'_> {
    fn clone(&self) -> Self {
        BoundingBox {
            orig: self.orig,
            len2: self.len2,
            objs: match &self.objs {
                None => None,
                Some(v) => Some(v.clone())
            },
            child_boxes: match &self.child_boxes {
                None => None,
                Some(v) => Some(v.clone())
            },
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


pub fn box_contains_triangle(orig: &Point, len2: f64, t: &Triangle) -> bool {

    if box_contains_point(orig, len2, &t.incenter) {
        return true;
    }

    if box_contains_point(orig, len2, &t.corners.0) ||
       box_contains_point(orig, len2, &t.corners.1) ||
       box_contains_point(orig, len2, &t.corners.2) {
        return true;
    }

    let ab_ray = Ray {
        orig: t.corners.0,
        dir: t.corners.1.sub(&t.corners.0)
    };
    let ab_p = ab_ray.nearest_point(orig);
    let ap1 = ab_p.sub(&ab_ray.orig);
    let ab_v = ap1.dot(&ab_ray.dir);
    let op = ab_p.sub(orig);
    if box_contains_point(orig, len2, &ab_p) &&
        ab_v >= 0. && ab_v < ab_ray.dir.len2() {
        return true;
    }

    let ac_ray = Ray {
        orig: t.corners.0,
        dir: t.corners.2.sub(&t.corners.0)
    };
    let ac_p = ac_ray.nearest_point(orig);
    let ap2 = ac_p.sub(&ac_ray.orig);
    let ac_v = ap2.dot(&ac_ray.dir);
    if box_contains_point(orig, len2, &ac_p) &&
        ac_v >= 0. && ac_v < ac_ray.dir.len2() {
        return true;
    }

    let bc_ray = Ray {
        orig: t.corners.1,
        dir: t.corners.2.sub(&t.corners.1)
    };
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
        let edge_ray = Ray {
            orig: points[edge.0],
            dir: points[edge.1].sub(&points[edge.0])
        };
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

pub fn build_bounding_box<'a>(objs: &'a Vec<CollisionObject>, orig: &Point, len2: f64, maxdepth: usize, minobjs: usize) -> BoundingBox<'a> {
    let mut refvec: Vec<&CollisionObject> = Vec::with_capacity(objs.len());
    refvec.extend(objs);
    build_bounding_box_helper(&refvec, orig, len2, 0, maxdepth, minobjs)
}

fn build_bounding_box_helper<'a>(objs: &Vec<&'a CollisionObject>, orig: &Point, len2: f64, depth: usize, maxdepth: usize, minobjs: usize) -> BoundingBox<'a> {

    let mut subobjs: Vec<&CollisionObject> = Vec::with_capacity(objs.len());
    for o in objs {
        match o {
            CollisionObject::Triangle(t) => {
                if box_contains_triangle(orig, len2, t) {
                    subobjs.push(o);
                }
            },
            _ => panic!("Only triangles supported")
        };
    }

    if subobjs.len() == 0 {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: None,
            child_boxes: None,
            depth: depth
        };
    } else if subobjs.len() < minobjs || depth >= maxdepth {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: Some(subobjs),
            child_boxes: None,
            depth: depth
        };
    }

    let mut subboxes: Vec<BoundingBox> = Vec::with_capacity(8);
    let newlen2 = len2 / 2.;
    for i in 0..8 {
        let xoff = if (i & 1) == 0 { -1.*newlen2 } else { newlen2 } ;
        let yoff = if (i & 2) == 0 { -1.*newlen2 } else { newlen2 } ;
        let zoff = if (i & 4) == 0 { -1.*newlen2 } else { newlen2 } ;
        let bbox = build_bounding_box_helper(&subobjs,
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
        objs: None,
        child_boxes: Some(subboxes),
        depth: depth
    }
}

pub fn build_trivial_bounding_box<'a>(objs: &'a Vec<CollisionObject>, orig: &Point, len2: f64) -> BoundingBox<'a> {

    let mut refvec: Vec<&CollisionObject> = Vec::with_capacity(objs.len());
    refvec.extend(objs);
    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: Some(refvec),
        child_boxes: None,
        depth: 0
    }
}

impl BoundingBox<'_> {
    
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
        let p = r.nearest_point(&self.orig);
        let op = p.sub(&self.orig);

        let hit = (op.0.abs() < self.len2 &&
                   op.1.abs() < self.len2 &&
                   op.2.abs() < self.len2) ||
                   self.collides_face(r, 0) ||
                   self.collides_face(r, 1) ||
                   self.collides_face(r, 2) ||
                   self.collides_face(r, 3) ||
                   self.collides_face(r, 4) ||
                   self.collides_face(r, 5);

        hit
    }

    fn get_all_objects_for_ray(&self, r: &Ray, path: &mut Vec<BoundingBox>) -> HashMap<usize, &CollisionObject> {

        let mut objmap: HashMap<usize, &CollisionObject> = HashMap::new();

        self.get_all_objects_for_ray_helper(r, &mut objmap, path);

        objmap
    }

    fn get_all_objects_for_ray_helper<'a>(&'a self, r: &Ray, objmap: &mut HashMap<usize, &'a CollisionObject>, path: &mut Vec<BoundingBox>) {
        if self.objs.is_none() &&
           self.child_boxes.is_none() {
            return;
        }
        if self.collides(r) {
            if self.objs.is_some() {
                objmap.extend(self.objs.as_ref().unwrap().iter().map(
                    |t| { (t.getid(), *t) }
                ));
            }

            if self.child_boxes.is_some() {
                for cbox in self.child_boxes.as_ref().unwrap() {
                    cbox.get_all_objects_for_ray_helper(r, objmap, path);
                }
            }
        }
    }

    pub fn print_tree(&self) {
        self.print();
        if self.child_boxes.is_some() {
            for b in self.child_boxes.as_ref().unwrap() {
                b.print_tree();
            }
        }
    }

    pub fn print(&self) {
        println!("BS: {} {:?} {} {} {}",
                 self.depth,
                 self.orig, self.len2,
                 match &self.objs {
                    None => -1,
                    Some(o) => o.len() as i64
                 },
                 match &self.child_boxes {
                    None => -1,
                    Some(b) => b.len() as i64
                 });
    }

    pub fn find_obj<'a>(&'a self, id: usize) -> Vec<&'a BoundingBox> {
        let mut found_list: Vec<&BoundingBox> = Vec::new();
        self.find_obj_helper(id, &mut found_list);
        found_list
    }

    pub fn find_obj_helper<'a>(&'a self, id: usize, found_list: &mut Vec<&BoundingBox<'a>>) {
        if self.objs.is_some() {
            for obj in self.objs.as_ref().unwrap() {
                if obj.getid() == id {
                    found_list.push(&self);
                }
            }
        }

        if self.child_boxes.is_some() {
            for b in self.child_boxes.as_ref().unwrap() {
                b.find_obj_helper(id, found_list);
            }
        }
    }

}

pub struct Scene<'a> {
    pub boxes: BoundingBox<'a>,
    pub otherobjs: Vec<CollisionObject>
}

fn color_ray(r: &Ray, s: &Scene, obj: &CollisionObject, point: &Point, face: &CollisionFace, depth: i64) -> Color {
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

fn project_ray(r: &Ray, s: &Scene, ignore_objid: usize, depth: i64) -> Color {

    if depth > 5 {
        return Vec3(0., 0., 0.);
    }

    let blue = Vec3(0.5, 0.7, 1.);

    let mut path: Vec<BoundingBox> = Vec::new();
    let mut objs = s.boxes.get_all_objects_for_ray(r, &mut path);
    objs.extend(s.otherobjs.iter().map(|o| (o.getid(), o)));

    let intersections: Vec<(f64, Point, CollisionFace, &CollisionObject)> = objs.iter().filter_map(
        |(k, s)| {
            if ignore_objid == *k {
                None
            } else {
                match s.intersects(&r) {
                    Some(p) => Some((p.0, p.1, p.2, *s)),
                    None    => None
                }
            }
        }).collect();

    if intersections.len() == 0 {
        blue
    } else {
        let (_dist, point, face, obj) = intersections.iter().fold(&intersections[0],
            |acc, x| {
                let (dist, _, _, _) = x;
                let (acc_dist, _, _, _) = acc;
                if dist < acc_dist { x } else { acc }
            });
            let t4 = time::Instant::now();
            color_ray(r, s, obj, point, &face, depth)

    }
}

#[derive(Clone, Copy,Debug)]
pub struct Viewport {
    width: usize,
    height: usize,

    orig: Point,
    cam: Point,
    
    vu: Vec3,
    vv: Vec3,

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

pub fn create_viewport(px: (u32, u32), size: (f64, f64), pos: &Point, dir: &Vec3, fov: f64, c_roll: f64, samples: usize) -> Viewport {

    let dist = size.0 / (2. * (fov.to_radians() / 2.).tan());

    let cam = *pos;

    let rot_basis = create_transform(dir, c_roll);

    // println!("YPR: {} {} {}", yaw, pitch, roll);

    // println!("Basis:");
    // println!(" {} {} {}", rot_basis.0.0, rot_basis.0.1, rot_basis.0.2);
    // println!(" {} {} {}", rot_basis.1.0, rot_basis.1.1, rot_basis.1.2);
    // println!(" {} {} {}", rot_basis.2.0, rot_basis.2.1, rot_basis.2.2);

    let orig = Vec3(1.*size.1/2., -1.*size.0/2., dist);
    let orig_r = cam.add(&orig.change_basis(rot_basis));

    // println!("orig: {} {} {}", orig.0, orig.1, orig.2);
    // println!("orig_r: {} {} {}", orig_r.0, orig_r.1, orig_r.2);

    let vu = Vec3(0., size.0, 0.);
    let vu_r = vu.change_basis(rot_basis);

    let vv = Vec3(-1. * size.1, 0., 0.);
    let vv_r = vv.change_basis(rot_basis);

    // println!("vu_r: {} {} {}", vu_r.0, vu_r.1, vu_r.2);
    // println!("vv_r: {} {} {}", vv_r.0, vv_r.1, vv_r.2);

    Viewport {
        width: px.0 as usize,
        height: px.1 as usize,
        orig: orig_r,
        cam: cam,
        vu: vu_r,
        vv: vv_r,
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

        Ray {
            orig: px_u,
            dir: px_u.sub(&self.cam).unit()
        }
    }

    fn walk_ray_set(&self, s: &Scene, data: & mut[Color], rows: Range<usize>,
                    t_num: usize, progress_tx: Sender<(usize, usize, usize)>) {

        let start_row = rows.start;
        let total_rays = rows.len()*self.width*self.samples_per_pixel;
        let mut rays_so_far = 0;
        for x in rows {
            for y in 0..self.width {
                let mut acc = Vec3(0., 0., 0.);
                for _i in 0..self.samples_per_pixel {
                    let ray_color = project_ray(&self.pixel_ray((x,y)), s, 0, 0);
                    acc = acc.add(&ray_color);

                    rays_so_far += 1;
                    if rays_so_far % 100 == 0 {
                        let _ = progress_tx.send((t_num, rays_so_far, total_rays));
                    }
                }
                data[((x-start_row)*self.width + y) as usize] = acc.mult(1./(self.samples_per_pixel as f64));

            }
        }
        let _ = progress_tx.send((t_num, total_rays, total_rays));
    }

    pub fn walk_rays(&self, s: &Scene, data: & mut[Color], threads: usize) -> ProgressCtx {

        let rows_per_thread = (self.height as usize) / threads;
        let mut data_list: Vec<(usize, Arc<Mutex<Vec<Color>>>)> = Vec::new();
        let (progress_tx, progress_rx) = channel();
        let total_rays = self.height * self.width * self.samples_per_pixel;

        let mut progress_io = progress::create_ctx(threads, total_rays, true);

        thread::scope(|sc| {
            for t in 0..threads {
                let rows = if t < (threads-1) {
                    rows_per_thread
                } else {
                    rows_per_thread + (self.height - rows_per_thread*threads)
                };
                let row_range = t*rows_per_thread..(t*rows_per_thread + rows);
                let alloc_size = rows * self.width;
                let data_mut: Arc<Mutex<Vec<Color>>> = Arc::new(Mutex::new(vec![Vec3(0., 0., 0.); alloc_size]));
                let t_data = Arc::clone(&data_mut);
                data_list.push((rows, data_mut));
                let t_progress_tx = progress_tx.clone();
                sc.spawn(move || {
                    let mut d = t_data.lock().unwrap();
                    self.walk_ray_set(s, &mut d[..], row_range, t, t_progress_tx);
                });
            }

            drop(progress_tx);
            'waitloop: loop {
                match progress_rx.recv() {
                    Ok((t_num, rays_so_far, total_rays)) => {
                        progress_io.update(t_num, rays_so_far, total_rays);
                    }
                    Err(x) => {
                        break 'waitloop;
                    }
                };
            }
        });

        let mut curr_row = 0;
        for (rows, d_mut) in data_list {
            let d = d_mut.lock().unwrap();
            data[curr_row*self.width..(curr_row+rows)*self.width].clone_from_slice(&d[..]);
            curr_row += rows;
        }

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
