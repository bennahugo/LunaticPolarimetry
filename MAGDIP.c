/* MAGDIP.f -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

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
/* Subroutine */ int magdip_aips(real *glat, real *glong, real *radius, real *h__)
{
    /* Initialized data */

    static real re = 6367650.f;

    /* System generated locals */
    doublereal d__1;

    /* Builtin functions */
    double cos(doublereal), sin(doublereal), sqrt(doublereal), acos(
	    doublereal), atan2(doublereal, doublereal);

    /* Local variables */
    static doublereal ca, cb;
    static real hd[3];
    static doublereal sa, sb;
    static real x0m, y0m, z0m, cla, clo, sla, slo;
    static doublereal pos0[3], pos1[3];
    static real hmag, fact;
    static doublereal post2[3], colat, raddip, londip, postmp[3];

/* ----------------------------------------------------------------------- */
/* ! Calculate Earth's magnetic field components */
/* # Util */
/* ----------------------------------------------------------------------- */
/* ;  Copyright (C) 1996 */
/* ;  Associated Universities, Inc. Washington DC, USA. */
/* ; */
/* ;  This program is free software; you can redistribute it and/or */
/* ;  modify it under the terms of the GNU General Public License as */
/* ;  published by the Free Software Foundation; either version 2 of */
/* ;  the License, or (at your option) any later version. */
/* ; */
/* ;  This program is distributed in the hope that it will be useful, */
/* ;  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* ;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* ;  GNU General Public License for more details. */
/* ; */
/* ;  You should have received a copy of the GNU General Public */
/* ;  License along with this program; if not, write to the Free */
/* ;  Software Foundation, Inc., 675 Massachusetts Ave, Cambridge, */
/* ;  MA 02139, USA. */
/* ; */
/* ;  Correspondence concerning AIPS should be addressed as follows: */
/* ;         Internet email: aipsmail@nrao.edu. */
/* ;         Postal address: AIPS Project Office */
/* ;                         National Radio Astronomy Observatory */
/* ;                         520 Edgemont Road */
/* ;                         Charlottesville, VA 22903-2475 USA */
/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/*   Routine to compute the earth's magnetic field strength from an */
/*   offset dipole model.  Adapted from Handbook of Geophysics and Space */
/*   Envirnoments (circa 1965) S. L. Valley ed. Air Force Cambridge */
/*   Research Laboratories and Chapman and Bartels, 1940, GEOPHYSICS, */
/*   Oxford) */
/*      NOTE: The Gaussian coefficients from Chapman and Bartels give */
/*   a slightly better representation of the field than Valley so these */
/*   values are used here. */
/*      Values of H returned are probably good to better than 20%. */
/*   At the VLA the model is 6% low in total intensity and 11 deg W in */
/*   magnetic declination. */
/*    Inputs: */
/*     GLAT    R    Geocentric latitude (radians) */
/*     GLONG   R    Geocentric EAST longitude (radians) */
/*     RADIUS  R    Distance from the center of the earth (m) */
/*    Output: */
/*     H(3)    R    Magnetic field vector (gauss), */
/*                  (1) = positive away from earth center, */
/*                  (2) = positive east, */
/*                  (3) = positive north. */
/* ----------------------------------------------------------------------- */
/*                                       Geographic coordinates of */
/*                                       North magnetic pole. */
/*                                       Gaussian coefficients(gauss): */
/*                                       From Handbook of Geophysics... */
/*                                       Epoch 1960. */
/*                                       Modified?????? */
/*      PARAMETER (G10 = -0.30509) */
/*      PARAMETER (G11 = -0.02181/2.0) */
/*      PARAMETER (G20 = -0.02196/2.0) */
/*      PARAMETER (G21 =  0.05145/3.0) */
/*      PARAMETER (G22 =  0.01448/4.0) */
/*      PARAMETER (H11 =  0.05841/2.0) */
/*      PARAMETER (H21 = -0.03443/3.0) */
/*      PARAMETER (H22 =  0.00172/4.0) */
/*                                       Chapman values Epoch 1922 */
/*                                       SQRT3 = sqrt (3.0) */
/*                                       Compute dipole center in units */
/*                                       of earth radius. */
/*                                       RE = Radius of earth (avg polar */
/*                                       and equitorial) */
    /* Parameter adjustments */
    --h__;

    /* Function Body */
/* ----------------------------------------------------------------------- */
/*                                       Center of dipole */
    x0m = -.050765871760315169f * re;
    y0m = re * .016744025199613304f;
    z0m = re * .0061112536050304533f;
/*                                       Convert to earth center x,y,z */
/*                                       Here y=> 90 e long. */
    pos0[0] = *radius * cos(*glat) * cos(*glong);
    pos0[1] = *radius * cos(*glat) * sin(*glong);
    pos0[2] = *radius * sin(*glat);
/*                                       Translate */
    postmp[0] = pos0[0] - x0m;
    postmp[1] = pos0[1] - y0m;
    postmp[2] = pos0[2] - z0m;
/*                                       Rotate to dipole coord. */
    ca = cos(5.0588368311250012f);
    sa = sin(5.0588368311250012f);
    cb = sin(1.372352389275f);
    sb = -cos(1.372352389275f);
    post2[0] = (postmp[0] * ca + postmp[1] * sa) * cb + postmp[2] * sb;
    post2[1] = postmp[1] * ca - postmp[0] * sa;
    post2[2] = postmp[2] * cb - sb * (postmp[0] * ca + postmp[1] * sa);
/*                                       Polar coordinates in dipole. */
    raddip = sqrt(post2[0] * post2[0] + post2[1] * post2[1] + post2[2] * 
	    post2[2]);
    colat = acos(post2[2] / raddip);
    londip = atan2(post2[1], post2[0]);
    cla = sin(colat);
    sla = cos(colat);
    clo = cos(londip);
    slo = sin(londip);
/*                                       Terms of dipole, local */
/* Computing 3rd power */
    d__1 = re / raddip;
    fact = sqrt(.099805649999999996f) * (d__1 * (d__1 * d__1));
    h__[1] = fact * -2.f * cos(colat);
    h__[2] = 0.f;
    h__[3] = fact * sin(colat);
/*                                       Rotate to dipole centered */
    hd[0] = (h__[1] * cla - h__[3] * sla) * clo - h__[2] * slo;
    hd[1] = h__[2] * clo + (h__[1] * cla - h__[3] * sla) * slo;
    hd[2] = h__[3] * cla + h__[1] * sla;
/*                                       Modulus of HD */
    hmag = sqrt(hd[0] * hd[0] + hd[1] * hd[1] + hd[2] * hd[2]);
/*                                       Find position 1 km from */
/*                                       position in the direction of HD. */
    post2[0] += hd[0] * 1e3f / hmag;
    post2[1] += hd[1] * 1e3f / hmag;
    post2[2] += hd[2] * 1e3f / hmag;
/*                                       Rotate new position to earth */
/*                                       system. */
    postmp[0] = (post2[0] * cb - post2[2] * sb) * ca - post2[1] * sa;
    postmp[1] = post2[1] * ca + (post2[0] * cb - post2[2] * sb) * sa;
    postmp[2] = post2[2] * cb + post2[0] * sb;
/*                                       Translate to earth center */
    pos1[0] = postmp[0] + x0m;
    pos1[1] = postmp[1] + y0m;
    pos1[2] = postmp[2] + z0m;
/*                                       Earth centered field */
    hd[0] = (pos1[0] - pos0[0]) * .001f * hmag;
    hd[1] = (pos1[1] - pos0[1]) * .001f * hmag;
    hd[2] = (pos1[2] - pos0[2]) * .001f * hmag;
/*                                       Earth local field */
    cla = cos(*glat);
    sla = sin(*glat);
    clo = cos(*glong);
    slo = sin(*glong);
    h__[1] = (hd[0] * clo + hd[1] * slo) * cla + hd[2] * sla;
    h__[2] = hd[1] * clo - hd[0] * slo;
    h__[3] = hd[2] * cla - (hd[0] * clo + hd[1] * slo) * sla;

/* L999: */
    return 0;
} /* magdip_ */

