#include <iostream>
#include <casacore/measures/Measures/EarthMagneticMachine.h>
#include <casacore/measures/Measures/EarthField.h>
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/casa/Quanta/MVTime.h>
#include <casacore/measures/Measures/MPosition.h>
#include <casacore/measures/Measures/MeasTable.h>
#include <casacore/measures/Measures/MDirection.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include "EMM/EMM_Model.h"

// External Functions
// Get the IGRF magnetic field around the Earth
extern "C" int jma_igrf13syn(
    double date,  // I  time (years AD, in UT) valid 1900--2025                                 
    double radius,// I  the position radius, in meters                    
    double clatit,// I  the cos(latitude, in radians)                    
    double slatit,// I  the sin(latitude, in radians)                    
    double clongi,// I  the cos(longitude, in radians)                   
    double slongi,// I  the sin(longitude, in radians) 
                  //    only geocentric coordinates are supported */       
    double *x,    // O  the magnetic field in the x direction     
    double *y,    // O  the magnetic field in the y direction             
    double *z__   // O  the magnetic field in the z direction 
                  //   the directions are x--local North on surface 
                  //                      y--local East on surface 
                  //                      z--local down 
    );

extern "C" int magdip_aips(
    float* glat, // latitude in radians
    float* glon, // longitude in radians
    float* radius,// I  the position radius, in meters
    float *xyz__        
                // note the direction switcharoo to igrf11 or igrf13         
    );

#if defined(__alpha__) || defined(__sparc64__) || defined(__x86_64__) || defined(__ia64__)
typedef int integer;
typedef unsigned int uinteger;
#else
typedef long int integer;
typedef unsigned long int uinteger;
#endif
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;

extern "C" int mgrf13_(integer *isv, doublereal *date, integer *itype, 
	doublereal *alt, doublereal *colat, doublereal *elong, doublereal *x, 
	doublereal *y, doublereal *z__, doublereal *f);

//https://gist.github.com/damithsj/c96a8482b282a3dc89bd
std::vector<double> linspace(double min, double max, int n)
{
    using namespace std;
	vector<double> result;
	// vector iterator
	int iterator = 0;

	for (int i = 0; i <= n-2; i++)	
	{
		double temp = min + i*(max-min)/(floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}

	//iterator += 1;

	result.insert(result.begin() + iterator, max);
	return result;
}

int main() {
    using namespace std;
    using namespace casacore;
    MEpoch epo(MVEpoch(MVTime(2020,1,1,1).day()));
    int numlons = 1024;
    int numlats = 512;
    std::vector<double> lons(linspace(-180 * M_PI / 180., +180  * M_PI / 180., numlons));
    std::vector<double> lats(linspace(-90  * M_PI / 180., +90  * M_PI / 180., numlats));
    double earthalt = 6371000;
    double ionoalt = 450e3;
    double refalt = ionoalt + earthalt;
    {
        ofstream casa_igrf12;
        casa_igrf12.open ("casa_igrf.txt");
        casa_igrf12 << "LON,LAT,STRENGTH(nT)" << endl;
        Quantity alt(refalt, "m");
        cout << "Writing CASA IGRFv12 field map" << endl;
        uint progress = 0;
        uint nbar = 0;
        for (auto ilon = lons.begin(); ilon != lons.end(); ++ilon) {
            Quantity lon(*ilon, "rad");
            for (auto ilat = lats.begin(); ilat != lats.end(); ++ilat) {
                Quantity lat(*ilat, "rad");
                MVPosition pos(alt, lon, lat);
                EarthField ef(EarthField::EarthFieldTypes::IGRF, epo.get("d").getValue());
                Vector<Double> xyz = ef(pos);
                casa_igrf12 << *ilon << "," << *ilat << "," << sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]) << endl;
            }
            int fprogress = int(floor(float(++progress) / 1024. * 100));
            if (fprogress >= nbar) {
                cout << fprogress << (nbar <= 90 ? "%..." : "%");
                nbar += 10;
            }
            cout.flush();
        }
        cout << endl;
        casa_igrf12.close();
    }
    {
        ofstream tony_igrf13;
        tony_igrf13.open ("tony_igrf.txt");
        tony_igrf13 << "LON,LAT,STRENGTH(nT)" << endl;
        cout << "Writing ALBUS IGRFv13 field map" << endl;
        uint progress = 0;
        uint nbar = 0;
        double alt = refalt;
        for (auto ilon = lons.begin(); ilon != lons.end(); ++ilon) {
            double clon = cos(*ilon), slon = sin(*ilon);
            for (auto ilat = lats.begin(); ilat != lats.end(); ++ilat) {
                double clat = cos(*ilat), slat = sin(*ilat);
                double x,y,z;
                jma_igrf13syn(2022,alt,clat,slat,clon,slon,&x,&y,&z);
                tony_igrf13 << *ilon << "," << *ilat << "," << sqrt(x*x + y*y + z*z) * 1e9 << endl;
            }
            int fprogress = int(floor(float(++progress) / 1024. * 100));
            if (fprogress >= nbar) {
                cout << fprogress << (nbar <= 90 ? "%..." : "%");
                nbar += 10;
            }
            cout.flush();
        }
        cout << endl;
        tony_igrf13.close();
    }
    {
        ofstream aips_magdip;
        aips_magdip.open ("aips_magdip.txt");
        aips_magdip << "LON,LAT,STRENGTH(nT)" << endl;
        cout << "Writing AIPS dipole MAGDIP field map" << endl;
        uint progress = 0;
        uint nbar = 0;
        float alt = refalt;
        for (auto ilon = lons.begin(); ilon != lons.end(); ++ilon) {
            float lon = (float)*ilon;
            for (auto ilat = lats.begin(); ilat != lats.end(); ++ilat) {
                float lat = (float)*ilat;
                float xyz[3];
                magdip_aips(&lat, &lon, &alt, &xyz[0]);
                aips_magdip << *ilon << "," << *ilat << "," << sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]) * 1e5 << endl;
            }
            int fprogress = int(floor(float(++progress) / 1024. * 100));
            if (fprogress >= nbar) {
                cout << fprogress << (nbar <= 90 ? "%..." : "%");
                nbar += 10;
            }
            cout.flush();
        }
        cout << endl;
        aips_magdip.close();
    }
    {
        ofstream aips_igrf13;
        aips_igrf13.open ("aips_igrf13.txt");
        aips_igrf13 << "LON,LAT,STRENGTH(nT)" << endl;
        cout << "Writing AIPS new IGRFv13 field map" << endl;
        uint progress = 0;
        uint nbar = 0;
        float alt = refalt;
        for (auto ilon = lons.begin(); ilon != lons.end(); ++ilon) {
            float lon = (float)*ilon;
            for (auto ilat = lats.begin(); ilat != lats.end(); ++ilat) {
                float lat = (float)*ilat;
                integer isv = 0;
                doublereal date = 2022.;
                integer itype = 2;
                doublereal altkm = alt * 1e-3;
                doublereal colat = M_PI - ((*ilat + M_PI / 2.) * 180 / M_PI);
                doublereal elon = *ilon  * 180 / M_PI;
                doublereal x,y,z,f;
                mgrf13_(&isv, &date, &itype, 
	                    &altkm, &colat, &elon, 
                        &x, &y, &z, &f);
                aips_igrf13 << *ilon << "," << *ilat << "," << sqrt(x*x + y*y + z*z) << endl;
            }
            int fprogress = int(floor(float(++progress) / 1024. * 100));
            if (fprogress >= nbar) {
                cout << fprogress << (nbar <= 90 ? "%..." : "%");
                nbar += 10;
            }
            cout.flush();
        }
        cout << endl;
        aips_igrf13.close();
    }
    {
        ofstream emm;
        emm.open ("rmextract_emm.txt");
        emm << "LON,LAT,STRENGTH(nT)" << endl;
        cout << "Writing RMExtract WMM field map" << endl;
        uint progress = 0;
        uint nbar = 0;
        float alt = refalt;
        // default
        WMM_Model model("/home/bhugo/workspace/casacore_igrf/EMM/WMM",2022., 0., 0., ionoalt / 1.0e3);
        for (auto ilon = lons.begin(); ilon != lons.end(); ++ilon) {
            float lon = (float)*ilon;
            for (auto ilat = lats.begin(); ilat != lats.end(); ++ilat) {
                float lat = (float)*ilat;
                model.setLonLat(lon * 180. / M_PI, lat * 180. / M_PI);
                //model.setLonLat(lon, lat);
                model.setEM();
                auto x = model.getX();
                auto y = model.getY();
                auto z = model.getZ();
                emm << *ilon << "," << *ilat << "," << sqrt(x*x + y*y + z*z) << endl;
            }
            int fprogress = int(floor(float(++progress) / float(numlons) * 100));
            if (fprogress >= nbar) {
                cout << fprogress << (nbar <= 90 ? "%..." : "%");
                nbar += 10;
            }
            cout.flush();
        }
        cout << endl;
        emm.close();
    }
    cout << "<done>" << endl;
}