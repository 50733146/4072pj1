//-----------------------------------------------------------------------------
// https://solvespace.com/forum.pl?action=viewthread&parent=3374&tt=1609280253
// Triangle mesh file reader. Reads an STL file triangle mesh and creates
// a SovleSpace SMesh from it. Supports only Linking, not import.
//
// Copyright 2020 Paul Kahler.
//-----------------------------------------------------------------------------
#include "solvespace.h"
#include "sketch.h"

#include<iostream>
#include<fstream>


// Make a new point - type doesn't matter since we will make a copy later
static hEntity newPoint(EntityList *el, int *id, Vector p) {
    Entity en = {};
    en.type = Entity::Type::POINT_N_COPY;
    en.extraPoints = 0;
    en.timesApplied = 0;
    en.group.v = 462;
    en.actPoint = p;
    en.construction = false;
    en.style.v = Style::DATUM;
    en.actVisible = true;
    en.forceHidden = false;

    *id = *id+1;
    en.h.v = *id + en.group.v*65536;    
    el->Add(&en);
    return en.h;
}

// check if a vertex is unique and add it via newPoint if it is.
static void addVertexIfUnique(float x, float y, float z, EntityList *el) {
    
    //uniqueness check
	Entity tEn={};

    int isNew=1;
    for(int i=0;i<el->n;i++)
    {
		tEn=el->Get(i);
        if (x== tEn.actPoint.x && y== tEn.actPoint.y && z== tEn.actPoint.z ) {
            isNew=0;
            break;
        }    
    }
    
    if (isNew==1) {
        int id = el->n+2;
        newPoint(el, &id, Vector::From(x,y,z));
    }  
}

static int StrStartsWith(const char *str, const char *start) {
	return memcmp(str, start, strlen(start)) == 0;
}

namespace SolveSpace {


bool LinkStl(const Platform::Path &filename, EntityList *el, SMesh *m, SShell *sh) {
    dbp("\nLink STL triangle mesh.");

	uint32_t n;
	uint32_t color;
	float x, y, z;
	float xn, yn, zn;
	std::string keyname, a1, b1;
	double x1, y1, z1;
	double x2, y2, z2;
	double x3, y3, z3;
	int numMember = 0;


	el->Clear();
    std::string data;
    if(!ReadFile(filename, &data)) {
        Error("Couldn't read from '%s'", filename.raw.c_str());
        return false;
    }
    
    std::stringstream f(data);

    char str[80] = {};
    f.read(str, 80);
	if (StrStartsWith(str, "solid")) {
		//solid
		std::string fn = filename.FileName();
		std::ifstream file(fn);
		while (file >> keyname) {
			STriangle tr = {};
			if (keyname == "facet") {
				//facet normal 0.2992115 0.012533324 0.0
				file >> a1 >> xn >> yn >> zn;
				//outer loop
				file >> a1 >> b1;
				//vertex 0.3 -7.347881E-17 0.49500003
				file >> a1 >> x1 >> y1 >> z1;
				//vertex 0.3 -7.347881E-17 -0.105
				file >> a1 >> x2 >> y2 >> z2;
				//vertex 0.29763442 0.037599973 - 0.105
				file >> a1 >> x3 >> y3 >> z3;
				//endloop
				file >> a1;
				//endfacet
				file >> a1;

				numMember = numMember + 1;

				tr.an = Vector::From(xn, yn, zn);
				tr.bn = tr.an;
				tr.cn = tr.an;

				tr.a.x = x1;
				tr.a.y = y1;
				tr.a.z = z1;

				tr.b.x = x2;
				tr.b.y = y2;
				tr.b.z = z2;

				tr.c.x = x3;
				tr.c.y = y3;
				tr.c.z = z3;

				m->AddTriangle(&tr);
			}

		}

	}

	else {

		f.read((char*)&n, 4);
		dbp("%d triangles", n);

		for (uint32_t i = 0; i < n; i++) {
			STriangle tr = {};

			// read the triangle normal
			f.read((char*)&xn, 4);
			f.read((char*)&yn, 4);
			f.read((char*)&zn, 4);
			tr.an = Vector::From(xn, yn, zn);
			tr.bn = tr.an;
			tr.cn = tr.an;

			f.read((char*)&x, 4);
			f.read((char*)&y, 4);
			f.read((char*)&z, 4);
			tr.a.x = x;
			tr.a.y = y;
			tr.a.z = z;
			addVertexIfUnique(x, y, z, el);

			f.read((char*)&x, 4);
			f.read((char*)&y, 4);
			f.read((char*)&z, 4);
			tr.b.x = x;
			tr.b.y = y;
			tr.b.z = z;
			addVertexIfUnique(x, y, z, el);

			f.read((char*)&x, 4);
			f.read((char*)&y, 4);
			f.read((char*)&z, 4);
			tr.c.x = x;
			tr.c.y = y;
			tr.c.z = z;
			addVertexIfUnique(x, y, z, el);

			f.read((char*)&color, 2);
			if (color & 0x8000) {
				tr.meta.color.red = (color >> 7) & 0xf8;
				tr.meta.color.green = (color >> 2) & 0xf8;
				tr.meta.color.blue = (color << 3);
				tr.meta.color.alpha = 255;
			}
			else {
				tr.meta.color.red = 90;
				tr.meta.color.green = 120;
				tr.meta.color.blue = 140;
				tr.meta.color.alpha = 255;
			}

			m->AddTriangle(&tr);
		}
	}
    SK.GetGroup(SS.GW.activeGroup)->forceToMesh = true;
    
    return true;
}



}
