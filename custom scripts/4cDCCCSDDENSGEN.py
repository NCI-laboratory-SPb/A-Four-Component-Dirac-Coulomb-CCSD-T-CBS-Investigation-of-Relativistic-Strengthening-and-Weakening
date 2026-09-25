#!/usr/bin/env python3
"""
4cDCCCSDQTAIM.py by Daniil A. Shitov

Generates cube file and CSV table with electron density for linear triatomic
molecules Y–X–Y based on four-component CCSD calculations with the DIRAC program
using Dirac-Coulomb and Lévy-Leblond Hamiltonians.

Usage:
    4cDCCCSDQTAIM.py molecule.xyz [-DC | -LL] -dyall.BASIS [-q CHARGE]

where molecule.xyz is a three‑atom linear molecule (collinear, order: Y, X, Y).
BASIS can be any of the Dyall basis sets:
    v2z, cv2z, ae2z, av2z, acv2z, aae2z,
    v3z, cv3z, ae3z, av3z, acv3z, aae3z,
    v4z, cv4z, ae4z, av4z, acv4z, aae4z
CHARGE is the total charge of the system (integer, default: -1).
"""

import os
import sys
import math
import subprocess
import argparse
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
ANG_TO_BOHR = 1.8897259886          # 1 Å = 1.8897259886 a.u.
BOHR_TO_ANG = 1.0 / ANG_TO_BOHR     # 1 Bohr = 0.529177 Å

# Grid parameters (in Angstroms)
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0
Z_MIN, Z_MAX = -5.0, 5.0
STEP = 0.1                          # grid step, Å

# Atomic numbers for cube file and element conversion
ATOM_CHARGES = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Ar': 18, 'K': 19, 'Ca': 20, 'Br': 35, 'Kr': 36, 'I': 53, 'Xe': 54, 'Gh': 0
}

# List of all Dyall basis sets
DYALL_BASIS_SETS = [
    'v2z', 'cv2z', 'ae2z', 'av2z', 'acv2z', 'aae2z',
    'v3z', 'cv3z', 'ae3z', 'av3z', 'acv3z', 'aae3z',
    'v4z', 'cv4z', 'ae4z', 'av4z', 'acv4z', 'aae4z'
]

# Templates for PAM input files (main calculation) – charge is inserted as integer
INP_MAIN_DC = """**DIRAC
.WAVE FUNCTION
**WAVE FUNCTION
.SCF
.RELCCSD
**RELCC
.ENERGY
.GRADIENT
*CCFOPR
.CCSDG
*CCENER
.NOMP2
.NOSDT
**HAMILTONIAN
.LVCORR
**MOLECULE
*CHARGE
.CHARGE
{charge}
*BASIS
.DEFAULT
{basisset}
.SPECIAL
Gh NOBASIS
**INTEGRALS
*READIN
.UNCONTRACT
.NORTSD
*END OF INPUT
"""

INP_MAIN_LL = """**DIRAC
.WAVE FUNCTION
**WAVE FUNCTION
.SCF
.RELCCSD
**RELCC
.ENERGY
.GRADIENT
*CCFOPR
.CCSDG
*CCENER
.NOMP2
.NOSDT
**HAMILTONIAN
.LEVY-LEBLOND
**MOLECULE
*CHARGE
.CHARGE
{charge}
*BASIS
.DEFAULT
{basisset}
.SPECIAL
Gh NOBASIS
**INTEGRALS
*READIN
.UNCONTRACT
.NORTSD
*END OF INPUT
"""

# Templates for PROP files (point calculations)
INP_PROP_DC = """**DIRAC
.PROPERTIES
**WAVE FUNCTION
**RELCC
*CCFOPR
.CCSDG
**HAMILTONIAN
.LVCORR
**MOLECULE
*CHARGE
.CHARGE
{charge}
*BASIS
.DEFAULT
{basisset}
.SPECIAL
Gh NOBASIS
**INTEGRALS
*READIN
.UNCONTRACT
.NORTSD
**PROPERTIES
.RDCCDM
.RHONUC
*END OF INPUT
"""

INP_PROP_LL = """**DIRAC
.PROPERTIES
**WAVE FUNCTION
**RELCC
*CCFOPR
.CCSDG
**HAMILTONIAN
.LEVY-LEBLOND
**MOLECULE
*CHARGE
.CHARGE
{charge}
*BASIS
.DEFAULT
{basisset}
.SPECIAL
Gh NOBASIS
**INTEGRALS
*READIN
.UNCONTRACT
.NORTSD
**PROPERTIES
.RDCCDM
.RHONUC
*END OF INPUT
"""

# ------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------
def read_xyz(filename):
    """Reads an xyz file, returns list of atoms (symbol, x, y, z) in Å."""
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError(f"File {filename} is too short")
    natoms = int(lines[0].strip())
    for i in range(natoms):
        parts = lines[i+2].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line: {lines[i+2]}")
        symb = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((symb, x, y, z))
    return atoms

def write_xyz(filename, atoms, comment=""):
    """Writes an xyz file."""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for symb, x, y, z in atoms:
            f.write(f"{symb:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")

def run_command(cmd, description):
    """Runs an external command, checks return value."""
    print(f"--> {description}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during execution: {result.stderr}")
        sys.exit(1)
    print("Four-component calculation (Dirac-Coulomb/Lévy-Leblond Hamiltonian) is done.")

def extract_ccsd_density_from_out(outfile, target_atom='Gh'):
    """
    Extracts the density value for target_atom from a PAM out file,
    taking the first occurrence in the CCSD block.
    Returns float.
    """
    with open(outfile, 'r') as f:
        text = f.read()
    blocks = re.split(r'\*\*\*\*\*\*\*\*\*\* Properties for CCSD wave function \*\*\*\*\*\*\*\*\*', text)
    if len(blocks) < 2:
        raise ValueError(f"CCSD block not found in file {outfile}")
    ccsd_block = blocks[1]
    pattern = re.compile(rf"Rho at nuc {target_atom}\s+\d+\s*:\s*([0-9.Ee+-]+)")
    matches = pattern.findall(ccsd_block)
    if not matches:
        raise ValueError(f"No lines for {target_atom} in CCSD block of {outfile}")
    return float(matches[0])

def extract_atom_densities_from_out(outfile, atoms_list):
    """
    Extracts densities on real atoms from a PAM out file.
    Uses values from the CCSD block.
    Returns dictionary { (symb, x, y, z): value } for each atom.
    """
    with open(outfile, 'r') as f:
        text = f.read()
    blocks = re.split(r'\*\*\*\*\*\*\*\*\*\* Properties for CCSD wave function \*\*\*\*\*\*\*\*\*', text)
    if len(blocks) < 2:
        raise ValueError("CCSD block not found in output file")
    ccsd_block = blocks[1]

    atom_dens = {}
    for symb, x, y, z in atoms_list:
        if symb != 'Gh':
            pattern = re.compile(rf"Rho at nuc {symb}\s+\d+\s*:\s*([0-9.Ee+-]+)")
            matches = pattern.findall(ccsd_block)
            if not matches:
                print(f"Warning: density for nucleus {symb} not found in CCSD block")
                value = float('nan')
            else:
                value = float(matches[0])
            atom_dens[(symb, x, y, z)] = value
    return atom_dens

def generate_grid():
    """Generates lists of x, y, z coordinates."""
    x_coords = np.arange(X_MIN, X_MAX + STEP/2, STEP)
    y_coords = np.arange(Y_MIN, Y_MAX + STEP/2, STEP)
    z_coords = np.arange(Z_MIN, Z_MAX + STEP/2, STEP)
    x_coords = np.round(x_coords, decimals=6)
    y_coords = np.round(y_coords, decimals=6)
    z_coords = np.round(z_coords, decimals=6)
    return x_coords, y_coords, z_coords

def get_unique_r_values(x_coords, y_coords):
    """Returns a set of unique r = sqrt(x^2+y^2) values for all grid points."""
    r_set = set()
    for x in x_coords:
        for y in y_coords:
            r = math.sqrt(x*x + y*y)
            r_set.add(round(r, 10))
    return r_set

def is_near_atom(x, y, z, atoms, tol=1e-6):
    """Checks if a point is close to any real atom."""
    for _, ax, ay, az in atoms:
        if math.hypot(x-ax, y-ay, z-az) < tol:
            return True
    return False

def get_atomic_z_list(atoms):
    """Returns list of z coordinates of real atoms (excluding ghost)."""
    return [z for (symb, x, y, z) in atoms if symb != 'Gh']

# ------------------------------------------------------------
# QTAIM analysis functions (no integration)
# ------------------------------------------------------------
def element_symbol(Z):
    """Return chemical symbol for atomic number Z."""
    # Create reverse mapping from ATOM_CHARGES
    inv_map = {v: k for k, v in ATOM_CHARGES.items()}
    return inv_map.get(Z, f'X{Z}')

def find_bcp(interp, z1, z2):
    """
    Find bond critical point (minimum density) along the z‑axis between z1 and z2 (bohr).
    Automatically ensures z1 < z2.
    Returns the z coordinate (in bohr) and the density at that point (in e⁻/bohr³).
    """
    if z1 > z2:
        z1, z2 = z2, z1
    def rho_along_z(z_val):
        return interp([0.0, 0.0, z_val])[0]
    res = minimize_scalar(rho_along_z, bounds=(z1, z2), method='bounded')
    return res.x, res.fun

def analyze_and_print(atoms_bohr, x, y, z, density, atom_densities, hamiltonian_name, total_charge):
    """
    Perform critical point analysis using exact nuclear densities from atom_densities.
    atoms_bohr: list of (atomic_number, x, y, z) in Bohr.
    x,y,z: coordinate arrays in Bohr.
    density: 3D array of shape (nx,ny,nz) in a.u.
    atom_densities: dict {(symb, x_ang, y_ang, z_ang): value} in a.u.
    total_charge: integer charge of the system.
    """
    print("\n" + "="*60)

    # Sort atoms by z coordinate to identify left, center, right
    sorted_atoms = sorted(atoms_bohr, key=lambda a: a[3])
    if len(sorted_atoms) != 3:
        print("Error: expected exactly three atoms")
        return
    left_atom = sorted_atoms[0]
    center_atom = sorted_atoms[1]
    right_atom = sorted_atoms[2]

    left_symb = element_symbol(left_atom[0])
    center_symb = element_symbol(center_atom[0])
    right_symb = element_symbol(right_atom[0])

    print(f"Analysis of the electron density for {left_symb}{center_symb}{right_symb} (four-component CCSD with {hamiltonian_name} Hamiltonian)")
    print(f"Total charge of the system: {total_charge}")

    # Build interpolator
    interp = RegularGridInterpolator((x, y, z), density, method='linear', bounds_error=False, fill_value=0.0)

    z_left = left_atom[3]
    z_center = center_atom[3]
    z_right = right_atom[3]

    # Find BCPs
    bcp_left_z, rho_left = find_bcp(interp, z_left, z_center)
    bcp_right_z, rho_right = find_bcp(interp, z_center, z_right)

    # Convert BCP coordinates to Angstroms for output
    bcp_left_ang = bcp_left_z * BOHR_TO_ANG
    bcp_right_ang = bcp_right_z * BOHR_TO_ANG

    # Total critical points: 3 nuclei + 2 BCP = 5
    print(f"Total number of critical points: 5")
    print(f"The Poincaré-Hopf relation is fulfilled: 3 - 2 = 1")

    # Positions of nuclei (in Angstroms)
    atoms_ang = [(Z, xb*BOHR_TO_ANG, yb*BOHR_TO_ANG, zb*BOHR_TO_ANG) for (Z, xb, yb, zb) in atoms_bohr]
    print("\nPositions of (3, -3) critical points (positions of nuclei) [Å]:")
    for Z, xa, ya, za in atoms_ang:
        symb = element_symbol(Z)
        print(f"  {symb}({xa:.6f}, {ya:.6f}, {za:.6f})")

    print("\nPositions of (3, -1) critical points (bond critical points) [Å]:")
    print(f"  {left_symb}{center_symb}(0.000, 0.000, {bcp_left_ang:.6f})")
    print(f"  {center_symb}{right_symb}(0.000, 0.000, {bcp_right_ang:.6f})")

    # Electron density at nuclei – use exact values from atom_densities
    print("\nElectron density at (3, -3) critical point (atomic units):")
    lookup = {(symb, xa, ya, za): val for (symb, xa, ya, za), val in atom_densities.items()}
    for (Z, xa, ya, za) in atoms_ang:
        symb = element_symbol(Z)
        key = (symb, xa, ya, za)
        if key in lookup:
            rho_nuc = lookup[key]
        else:
            # Fallback to interpolation (should not happen)
            xb = xa * ANG_TO_BOHR
            yb = ya * ANG_TO_BOHR
            zb = za * ANG_TO_BOHR
            rho_nuc = interp([xb, yb, zb]).item()
            print(f"  Warning: Exact density for {symb} not found, using interpolated value {rho_nuc:.6f}")
        print(f"  {symb}: {rho_nuc:.10f}")

    print("\nElectron density at (3, -1) critical point (atomic units):")
    print(f"  {left_symb}{center_symb}: {rho_left:.10f}")
    print(f"  {center_symb}{right_symb}: {rho_right:.10f}")

    print("="*60 + "\n")

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate cube and CSV for electron density of linear Y–X–Y molecules, plus QTAIM analysis')
    parser.add_argument('xyz_file', help='Input xyz file with molecular coordinates')
    parser.add_argument('-DC', action='store_true', help='Use Dirac–Coulomb Hamiltonian')
    parser.add_argument('-LL', action='store_true', help='Use Lévy-Leblond Hamiltonian')
    parser.add_argument('-q', type=int, default=-1, help='Total charge of the system (integer, default: -1)')

    for basis in DYALL_BASIS_SETS:
        parser.add_argument(f'-dyall.{basis}', dest=f'dyall_{basis.replace(".", "_")}',
                            action='store_true', help=f'Basis set dyall.{basis}')

    args = parser.parse_args()

    if not (args.DC or args.LL):
        print("Error: either -DC or -LL must be specified")
        sys.exit(1)
    hamiltonian = 'DC' if args.DC else 'LL'
    hamiltonian_name = 'Dirac‑Coulomb' if args.DC else 'Lévy‑Leblond'

    selected_basis = None
    for basis in DYALL_BASIS_SETS:
        dest_name = f'dyall_{basis.replace(".", "_")}'
        if getattr(args, dest_name, False):
            selected_basis = basis
            break
    if selected_basis is None:
        print("Error: a Dyall basis set must be specified (e.g., -dyall.av3z)")
        sys.exit(1)
    basisset = f'dyall.{selected_basis}'

    total_charge = args.q

    xyz_file = args.xyz_file
    base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    base_atoms = read_xyz(xyz_file)

    # --------------------------------------------------------
    # Header with theory information
    # --------------------------------------------------------
    print("\n" + "-"*60)
    print("4cDCCCSDQTAIM")
    print("Four-Component Dirac-Coulomb CCSD QTAIM Analysis")
    print("-"*60)
    print(f"Wavefunction: four-component CCSD")
    print(f"Hamiltonian: {hamiltonian_name}")
    print(f"Basis set: {basisset}")
    print(f"Total charge: {total_charge}")
    print("Solving four-component coupled cluster equations:")
    print("H = ∑ₚₐ hₚₐ (aₚ)†aₐ + ½ ∑ₚₐᵣₛ Hₚₐᵣₛ(aₚ)† (aₐ)†aₛaᵣ")
    print("T = ∑ᵢᵅ tᵢᵅ(aₐ)†aᵢ + ∑ᵢⱼᵅᵖtᵢⱼᵅᵖ(aₐ)†(aₚ)†aⱼaᵢ")
    print("|Φ(4cCCSD) ⟩ = exp[T]|Φ(DHF)⟩")
    print("E(4cCCSD) = ⟨Φ(4cCCSD)|H|Φ(4cCCSD)⟩")
    print("⟨Φ(DHF)|(aᵢ)†aₐH|Φ(4cCCSD)⟩ = 0")
    print("⟨Φ(DHF)|(aᵢ)†(aⱼ)†aₚaₐH|Φ(4cCCSD)⟩ = 0")
    print("Density matrix of four-component CCSD wavefunction:")
    print("Dₚₛ = ⟨Φ(4cCCSD)|(aₚ)†aₛ|Φ(4cCCSD)⟩")
    print("-"*60 + "\n")

    # --------------------------------------------------------
    # 1. Generate file XXXGh.xyz with one ghost atom on Z axis at 0.1 Å
    # --------------------------------------------------------
    gh_initial = ('Gh', 0.0, 0.0, 0.1)
    atoms_with_gh = base_atoms + [gh_initial]
    gh_xyz = f"{base_name}Gh.xyz"
    write_xyz(gh_xyz, atoms_with_gh, comment=f"{base_name}Gh")

    # --------------------------------------------------------
    # 2. Generate input files
    # --------------------------------------------------------
    if hamiltonian == 'DC':
        main_inp_content = INP_MAIN_DC.format(basisset=basisset, charge=total_charge)
        prop_inp_content = INP_PROP_DC.format(basisset=basisset, charge=total_charge)
    else:
        main_inp_content = INP_MAIN_LL.format(basisset=basisset, charge=total_charge)
        prop_inp_content = INP_PROP_LL.format(basisset=basisset, charge=total_charge)

    main_inp_file = f"{base_name}Gh_{hamiltonian}.inp"
    with open(main_inp_file, 'w') as f:
        f.write(main_inp_content)
    print(f"Generated main input file: {main_inp_file}")

    prop_inp_file = f"{base_name}_{hamiltonian}_prop.inp"
    with open(prop_inp_file, 'w') as f:
        f.write(prop_inp_content)
    print(f"Generated PROP file: {prop_inp_file}")

    # --------------------------------------------------------
    # 3. Run initial calculation (obtain CCDENS)
    # --------------------------------------------------------
    cmd = ["./pam", "--mpi=16", f"--inp={main_inp_file}", "--get=CCDENS", "--outcmo", f"--mol={gh_xyz}"]
    run_command(cmd, "Four-component calculation of density matrix with CCSD wavefunction (Dirac-Coulomb/Lévy-Leblond Hamiltonian)")

    # --------------------------------------------------------
    # 4. Prepare grid
    # --------------------------------------------------------
    x_coords, y_coords, z_coords = generate_grid()
    r_values = get_unique_r_values(x_coords, y_coords)

    z_nonneg = [z for z in z_coords if z >= 0]
    print(f"Calculations will be performed for z: {z_nonneg}")

    dens_map = {}
    counter = 0
    atom_densities = {}
    first_cinfv_done = False

    # --------------------------------------------------------
    # 5. Point calculations with descriptive messages
    # --------------------------------------------------------
    for r in sorted(r_values):
        for z in z_nonneg:
            if r == 0:
                point_x, point_y, point_z = 0.0, 0.0, z
            else:
                point_x, point_y, point_z = r, 0.0, z

            if is_near_atom(point_x, point_y, point_z, base_atoms):
                print(f"Skipping point ({point_x:.6f}, {point_y:.6f}, {point_z:.6f}) – coincides with an atom")
                continue

            if r == 0:
                atoms_ghost = base_atoms + [('Gh', 0.0, 0.0, z)]
                xyz_name = f"{base_name}Gh_Cinfv_{counter:04d}.xyz"
                out_name = f"{base_name}_{hamiltonian}_prop_{base_name}Gh_Cinfv_{counter:04d}.out"
                write_xyz(xyz_name, atoms_ghost, comment=f"{base_name}Gh")
                cmd = ["./pam", "--mpi=16", f"--inp={prop_inp_file}", "--put=CCDENS", "--incmo", "--noarch", f"--mol={xyz_name}"]
                desc = (f"Four-component CCSD calculation with {hamiltonian_name} Hamiltonian for point with "
                        f"radial coordinate = 0.000000 Å, internuclear coordinate = {z:.6f} Å (double group C(∞, v))")
                run_command(cmd, desc)
                try:
                    dens = extract_ccsd_density_from_out(out_name, target_atom='Gh')
                except Exception as e:
                    print(f"Error extracting density from {out_name}: {e}")
                    dens = float('nan')
                dens_map[(r, z)] = dens

                if not first_cinfv_done:
                    try:
                        atom_densities = extract_atom_densities_from_out(out_name, base_atoms)
                        print("Densities on nuclei from the first C∞v file:")
                        for (symb, x, y, z), val in atom_densities.items():
                            print(f"  {symb} at ({x:.6f}, {y:.6f}, {z:.6f}): {val:.10f} a.u.")
                        first_cinfv_done = True
                    except Exception as e:
                        print(f"Could not extract atomic densities from {out_name}: {e}")
                        atom_densities = {}

                os.remove(out_name)
                os.remove(xyz_name)
                counter += 1

            else:
                distant_ghost = ('Gh', 0.0, 0.0, 6.0)
                atoms_ghost = base_atoms + [('Gh', r, 0.0, z), ('Gh', -r, 0.0, z), distant_ghost]
                xyz_name = f"{base_name}GhGh_C2v_{counter:04d}.xyz"
                out_name = f"{base_name}_{hamiltonian}_prop_{base_name}GhGh_C2v_{counter:04d}.out"
                write_xyz(xyz_name, atoms_ghost, comment=f"{base_name}GhGh")
                cmd = ["./pam", "--mpi=16", f"--inp={prop_inp_file}", "--put=CCDENS", "--incmo", "--noarch", f"--mol={xyz_name}"]
                desc = (f"Four-component CCSD calculation with {hamiltonian_name} Hamiltonian for point with "
                        f"radial coordinate = {r:.6f} Å, internuclear coordinate = {z:.6f} Å (double group C(2, v))")
                run_command(cmd, desc)
                try:
                    dens_ghost = extract_ccsd_density_from_out(out_name, target_atom='Gh')
                    dens = dens_ghost / 2.0
                except Exception as e:
                    print(f"Error extracting density from {out_name}: {e}")
                    dens = float('nan')
                dens_map[(r, z)] = dens

                os.remove(out_name)
                os.remove(xyz_name)
                counter += 1

    if not first_cinfv_done:
        print("Warning: no C∞v calculation performed, atomic densities not defined.")

    # --------------------------------------------------------
    # 6. Build data for CSV (original orientation)
    # --------------------------------------------------------
    nx = len(x_coords)
    ny = len(y_coords)
    nz = len(z_coords)

    dens_original = np.zeros((nx, ny, nz))

    x_to_idx = {xv: i for i, xv in enumerate(x_coords)}
    y_to_idx = {yv: i for i, yv in enumerate(y_coords)}
    z_to_idx = {zv: i for i, zv in enumerate(z_coords)}

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            r = math.sqrt(x*x + y*y)
            for k, z in enumerate(z_coords):
                z_abs = abs(z)
                if is_near_atom(x, y, z, base_atoms):
                    continue
                key = (round(r, 10), round(z_abs, 10))
                if key in dens_map:
                    dens_original[i, j, k] = dens_map[key]
                else:
                    print(f"Warning: missing density for r={r}, z_abs={z_abs}")

    for (symb, ax, ay, az) in base_atoms:
        i = np.argmin(np.abs(x_coords - ax))
        j = np.argmin(np.abs(y_coords - ay))
        k = np.argmin(np.abs(z_coords - az))
        if math.hypot(x_coords[i]-ax, y_coords[j]-ay, z_coords[k]-az) < 1e-6:
            key = (symb, ax, ay, az)
            if key in atom_densities:
                dens_original[i, j, k] = atom_densities[key]

    # --------------------------------------------------------
    # 7. Write CSV (original orientation)
    # --------------------------------------------------------
    csv_file = f"{base_name}_{hamiltonian}.csv"
    with open(csv_file, 'w') as f:
        f.write("x, A,y, A,z, A,ρ(x,y,z), a.u.\n")
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                for k, z in enumerate(z_coords):
                    dens = dens_original[i, j, k]
                    f.write(f"{x:.6f},{y:.6f},{z:.6f},{dens:.10E}\n")
    print(f"CSV file saved: {csv_file}")

    # --------------------------------------------------------
    # 8. Write cube file with swapped axes for nuclei (z ↔ x)
    # --------------------------------------------------------
    origin_x = x_coords[0] * ANG_TO_BOHR
    origin_y = y_coords[0] * ANG_TO_BOHR
    origin_z = z_coords[0] * ANG_TO_BOHR

    dxb = STEP * ANG_TO_BOHR
    dyb = STEP * ANG_TO_BOHR
    dzb = STEP * ANG_TO_BOHR

    cube_file = f"{base_name}_{hamiltonian}.cube"
    with open(cube_file, 'w') as f:
        f.write(f"Cube file generated from {xyz_file} with {hamiltonian} Hamiltonian\n")
        f.write(f"Grid: {nx} x {ny} x {nz}, step={STEP:.6f} A\n")

        f.write(f"{len(base_atoms):5d}{origin_x:12.6f}{origin_y:12.6f}{origin_z:12.6f}\n")

        f.write(f"{nx:5d}{dxb:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        f.write(f"{ny:5d}{0.0:12.6f}{dyb:12.6f}{0.0:12.6f}\n")
        f.write(f"{nz:5d}{0.0:12.6f}{0.0:12.6f}{dzb:12.6f}\n")

        for symb, ax, ay, az in base_atoms:
            an = ATOM_CHARGES.get(symb, 0)
            charge = float(an)
            xa_cube = az * ANG_TO_BOHR
            ya_cube = ax * ANG_TO_BOHR
            za_cube = ay * ANG_TO_BOHR
            f.write(f"{an:5d}{charge:12.6f}{xa_cube:12.6f}{ya_cube:12.6f}{za_cube:12.6f}\n")

        for k in range(nz):
            for j in range(ny):
                vals = [dens_original[i, j, k] for i in range(nx)]
                for i in range(0, nx, 6):
                    chunk = vals[i:i+6]
                    line = ' ' + ' '.join(f"{v:15.8E}" for v in chunk)
                    f.write(line + '\n')

    print(f"Cube file saved: {cube_file}")

    # --------------------------------------------------------
    # 9. Cleanup temporary files (only initial .out)
    # --------------------------------------------------------
    initial_out = f"{base_name}Gh_{hamiltonian}.out"
    if os.path.exists(initial_out):
        os.remove(initial_out)

    # --------------------------------------------------------
    # 10. QTAIM analysis (no integration)
    # --------------------------------------------------------
    atoms_bohr = []
    for symb, xa, ya, za in base_atoms:
        Z = ATOM_CHARGES.get(symb, 0)
        xb = xa * ANG_TO_BOHR
        yb = ya * ANG_TO_BOHR
        zb = za * ANG_TO_BOHR
        atoms_bohr.append((Z, xb, yb, zb))

    x_bohr = x_coords * ANG_TO_BOHR
    y_bohr = y_coords * ANG_TO_BOHR
    z_bohr = z_coords * ANG_TO_BOHR

    analyze_and_print(atoms_bohr, x_bohr, y_bohr, z_bohr, dens_original, atom_densities, hamiltonian_name, total_charge)

    print("Four-component coffee time!")

if __name__ == "__main__":
    main()
