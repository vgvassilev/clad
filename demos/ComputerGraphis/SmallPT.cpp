//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to use clad in simple path tracer.
//
// author:  Alexander Penev <alexander_penev-at-yahoo.com>
//          Based on smallpt, a Path Tracer by Kevin Beason, 2008
//----------------------------------------------------------------------------//

// To compile the demo please type:
// path/to/clang -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so -I../../include/ -x c++ -std=c++11 -lstdc++ SmallPT.cpp \
// -o SmallPT
//
// To run the demo please type:
// ./SmallPT 5000 && xv image.ppm

// A typical invocation would be:
// ../../../../../obj/Debug+Asserts/bin/clang -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../../obj/Debug+Asserts/lib/libclad.dylib \
// -I../../include/ -x c++ -std=c++11 -lstdc++ SmallPT.cpp -o SmallPT
// ./SmallPT 5000 && xv image.ppm

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Necessary for clad to work include
//#include "clad/Differentiator/Differentiator.h"

struct Vec {
  float x, y, z; // position, also color (r,g,b)

  Vec(float x_=0, float y_=0, float z_=0) { x=x_; y=y_; z=z_; }

  Vec operator+(const Vec &b) const { return Vec(x+b.x, y+b.y, z+b.z); }
  Vec operator-(const Vec &b) const { return Vec(x-b.x, y-b.y, z-b.z); }
  Vec operator*(float b) const { return Vec(x*b, y*b, z*b); }
  float operator*(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // dot
  //float dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; }
  Vec operator%(const Vec &b) { return Vec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); } // cross
  Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
  Vec& norm() { return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
};

struct Ray {
  Vec o, d; // Origin and direction

  Ray(Vec o_, Vec d_): o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

float difftest(float x) {
  return 0;
}

struct Solid {
  float rad;   // radius
  Vec p; // position
  Vec e, c; // emission, color
  Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)

  Solid(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}

  float implicit(const Vec &x) const {
    //return x.dot(x)-rad*rad;
    //return x*x-rad*rad;
    return 0;
  }

  float implicit1(float x, float y, float z) const {
    return 0;
  }

  // returns distance, 0 if nohit
  float intersect(const Ray &r) const {
    // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    Vec op = p-r.o;
    ///float t, eps=1e-1, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
    float t, eps=1e-1, b=op*r.d, det=b*b-op*op+rad*rad;
    if (det<0) return 0; else det=sqrt(det);
    return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
  }

  // returns normal vector to surface in point x
  Vec normal(const Vec &x) const {
    return (x-p).norm();
  }
};

// Scene: radius, position, emission, color, material
Solid scene[] = {
  Solid(1e5,  Vec( 1e5+1,40.8,81.6),  Vec(), Vec(.75,.25,.25), DIFF), // Left
  Solid(1e5,  Vec(-1e5+99,40.8,81.6), Vec(), Vec(.25,.25,.75), DIFF), // Rght
  Solid(1e5,  Vec(50,40.8, 1e5),      Vec(), Vec(.75,.75,.75), DIFF), // Back
  Solid(1e5,  Vec(50,40.8,-1e5+170),  Vec(), Vec(),            DIFF), // Frnt
  Solid(1e5,  Vec(50, 1e5, 81.6),     Vec(), Vec(.75,.75,.75), DIFF), // Botm
  Solid(1e5,  Vec(50,-1e5+81.6,81.6), Vec(), Vec(.75,.75,.75), DIFF), // Top
  Solid(16.5, Vec(27,16.5,47),        Vec(), Vec(1,1,1)*.999,  SPEC), // Mirr
  Solid(16.5, Vec(73,16.5,78),        Vec(), Vec(1,1,1)*.999,  REFR), // Glas
  Solid(600,  Vec(50,681.6-.27,81.6), Vec(12,12,12), Vec(),    DIFF)  // Lite
};

inline float clamp(float x) {
  return x<0 ? 0 : x>1 ? 1 : x;
}

inline int toInt(float x) {
  return int(pow(clamp(x),1/2.2)*255+.5);
}

inline bool intersect(const Ray &r, float &t, int &id) {
  int n = sizeof(scene)/sizeof(scene[0]);
  float d, inf=t=1e20;

  for(int i=n; i--; )
    if ((d=scene[i].intersect(r))&&d<t) { t=d; id=i; }

  return t<inf;
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
  float t;  // distance to intersection
  int id=0; // id of intersected object

  // if miss, return black
  if (!intersect(r, t, id)) return Vec();

  // the hit object
  const Solid &obj = scene[id];

  // calculate intersection point
  Vec x=r.o+r.d*t;

  // calculate surface normal vector in point x
  Vec n=obj.normal(x);
  Vec nl=n*r.d<0 ? n : n*-1;

  // object base color
  Vec f=obj.c;
  float p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl

  if (++depth>5) {
    if (erand48(Xi)<p) f=f*(1/p); else return obj.e;
  } // R.R.

  if (obj.refl == DIFF) { // Ideal DIFFUSE reflection
    float r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
    Vec w=nl, u=((fabs(w.x)>.1 ? Vec(0,1) : Vec(1))%w).norm(), v=w%u;
    Vec d = (u*cos(r1)*r2s+v*sin(r1)*r2s+w*sqrt(1-r2)).norm();
    return obj.e + f.mult(radiance(Ray(x,d), depth, Xi));
  } else if (obj.refl == SPEC) { // Ideal SPECULAR reflection
    return obj.e + f.mult(radiance(Ray(x,r.d-n*2*(n*r.d)), depth, Xi));
  }

  // Ideal dielectric REFRACTION
  Ray reflRay(x, r.d-n*2*(n*r.d));
  bool into = n*nl>0; // Ray from outside going in?
  float nc=1, nt=1.5, nnt=into ? nc/nt : nt/nc, ddn=r.d*nl, cos2t;

  if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0) // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, Xi));

  Vec tdir = (r.d*nnt-n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
  float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir*n);
  float Re=R0+(1-R0)*c*c*c*c*c, Tr=1-Re, P=.25+.5*Re, RP=Re/P, TP=Tr/(1-P);

  return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette
    radiance(reflRay, depth, Xi)*RP : radiance(Ray(x,tdir), depth, Xi)*TP) :
    radiance(reflRay, depth, Xi)*Re+radiance(Ray(x,tdir), depth, Xi)*Tr);
}

int main(int argc, char *argv[]) {

  // Diff test
  //auto cladDiff = clad::differentiate(difftest, 1);
  //auto cladDiff = clad::differentiate(Solid::implicit1, 1);


  int w=1024, h=768, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples

  Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir
  Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r;
  Vec *frame=new Vec[w*h];

#pragma omp parallel for schedule(dynamic, 1) private(r)
  for (unsigned short y=0; y<h; y++) { // Loop over image rows
    //fprintf(stderr, "\nRendering (%d spp) %5.2f%%", samps*4, 100.*y/(h-1));
    for (unsigned short x=0, Xi[3]={0,0,(unsigned short)(y*y*y)}; x<w; x++) { // Loop cols
      for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++) { // 2x2 subpixel rows
        for (int sx=0; sx<2; sx++, r=Vec()) {     // 2x2 subpixel cols
          for (int s=0; s<samps; s++) {
fprintf(stderr, "\ny=%d, x=%d, sy=%d, sx=%d, s=%d, i=%d", y, x, sy, sx, s, i);
            float r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1 : 1-sqrt(2-r1);
            float r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1 : 1-sqrt(2-r2);
            Vec d = cx*(((sx+.5+dx)/2+x)/w-.5)+cy*(((sy+.5+dy)/2+y)/h-.5)+cam.d;
            r = r + radiance(Ray(cam.o+d*140, d.norm()), 0, Xi)*(1./samps);
          } // Camera rays are pushed ^^^^^ forward to start in interior
          frame[i] = frame[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
        }
      }
    }
  }

  // Write image to PPM file.
  FILE *f = fopen("image.ppm", "wb");
  fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
  for (unsigned short i=0; i<w*h; i++)
    fprintf(f, "%c%c%c", toInt(frame[i].x), toInt(frame[i].y), toInt(frame[i].z));
}
