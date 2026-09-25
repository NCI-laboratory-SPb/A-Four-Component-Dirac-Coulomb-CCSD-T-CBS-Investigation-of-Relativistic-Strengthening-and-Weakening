#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4cDCCCSDCBSRovibConstants.py - Calculate vibrational frequencies for YXY triatomic molecules
using two-point CBS extrapolation (dyall.av2z and dyall.av3z).

Usage:
  4cDCCCSDCBSRovibConstants.py input.xyz -DC|-LL -q CHARGE [options]

Options:
  -DC, -LL          Hamiltonian
  -q CHARGE         molecular charge (integer, default 0)
  --keep            keep all intermediate files (DIRAC inputs, outputs)
"""

import os
import sys
import numpy as np
import subprocess
import re
import glob
from argparse import ArgumentParser
from scipy.linalg import eigh_tridiagonal

# ---------- Constants ----------
bohr_to_Ang = 0.529177249
hartree_to_joule = 4.3597482e-18
planck_constant = 6.625e-34
speed_of_light = 299792458
mass_proton = 1836.152673425606
mass_proton_SI = 1.6726219236951e-27

# ---------- Atomic masses ----------
atomic_mass = {
    'H':1.00794, 'He':4.0026, 'Li':6.941, 'Be':9.0122, 'B':10.811,
    'C':12.0107, 'N':14.0067, 'O':15.9994, 'F':18.9984, 'Ne':20.1797,
    'Na':22.9898, 'Mg':24.3050, 'Al':26.9815, 'Si':28.0855, 'P':30.9738,
    'S':32.065, 'Cl':35.453, 'Ar':39.948, 'K':39.0983, 'Ca':40.078,
    'Br':79.904, 'I':126.9045, 'Kr':83.798, 'Hg':200.592, 'Cd':112.414,
    'Xe':131.293, 'Pb':207.2, 'Se':78.96, 'Sn':118.71, 'Bi':208.98, 'As':74.9216, 'Te':127.6
}

def get_mass(sym):
    return atomic_mass.get(sym, 1.0)

def parse_xyz_triatomic(xyz_file):
    with open(xyz_file) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    if natoms != 3:
        raise ValueError("Only linear triatomic molecules YXY are supported.")
    atoms = []
    coords = []
    for i in range(2, 5):
        parts = lines[i].split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    if atoms[0] != atoms[2]:
        raise ValueError("Molecule must be symmetric YXY.")
    for c in coords:
        if abs(c[0]) > 1e-6 or abs(c[1]) > 1e-6:
            raise ValueError("Molecule must be linear along Z axis.")
    z1 = coords[0][2]
    z2 = coords[1][2]
    z3 = coords[2][2]
    if abs(z2) > 1e-6:
        raise ValueError("Central atom must be at origin (z=0).")
    if abs(z1 + z3) > 1e-6:
        raise ValueError("Terminal atoms must be symmetric: z1 = -z3.")
    return atoms[0], atoms[1], abs(z1)

def generate_xyz_symmetric(Y, X, zY, outfile):
    with open(outfile, 'w') as f:
        f.write("3\n")
        f.write(f"{Y}-{X}-{Y}\n")
        f.write(f"{Y} 0.00000 0.00000 {zY:.8f}\n")
        f.write(f"{X} 0.00000 0.00000 0.00000\n")
        f.write(f"{Y} 0.00000 0.00000 -{zY:.8f}\n")

def generate_xyz_bending(Y, X, Z_eq, d, mY, mX, outfile):
    shiftX = - (mX * d) / (2 * mY)
    with open(outfile, 'w') as f:
        f.write("3\n")
        f.write(f"{Y}-{X}-{Y}\n")
        f.write(f"{Y} {shiftX:.8f} 0.00000 {Z_eq:.8f}\n")
        f.write(f"{X} {d:.8f} 0.00000 0.00000\n")
        f.write(f"{Y} {shiftX:.8f} 0.00000 -{Z_eq:.8f}\n")

def generate_xyz_antisymmetric(Y, X, Z_eq, d, mY, mX, outfile):
    shiftZ = (mX * d) / (2 * mY)
    z1 = Z_eq - shiftZ
    z2 = -Z_eq - shiftZ
    with open(outfile, 'w') as f:
        f.write("3\n")
        f.write(f"{Y}-{X}-{Y}\n")
        f.write(f"{Y} 0.00000 0.00000 {z1:.8f}\n")
        f.write(f"{X} 0.00000 0.00000 {d:.8f}\n")
        f.write(f"{Y} 0.00000 0.00000 {z2:.8f}\n")

def write_dirac_input_triatomic(mol_name, hamiltonian, basis, q, inp_file):
    ham_key = '.LVCORR' if hamiltonian == 'DC' else '.LEVY-LEBLOND'
    content = f"""**DIRAC
.WAVE FUNCTION
**WAVE FUNCTION
.SCF
.RELCCSD
**RELCC
.ENERGY
**HAMILTONIAN
{ham_key}
**MOLECULE
*CHARGE
.CHARGE
{q}
*BASIS
.DEFAULT
dyall.{basis}
**INTEGRALS
*READIN
.UNCONTRACT
*END OF INPUT
"""
    with open(inp_file, 'w') as f:
        f.write(content)

def run_dirac(inp_file, mol_file, work_dir):
    pam_path = os.path.abspath('./pam')
    if not os.path.isfile(pam_path):
        raise FileNotFoundError("pam executable not found in current directory (./pam)")
    cmd = f"{pam_path} --mpi=16 --inp={inp_file} --noarch --mol={mol_file}"
    try:
        subprocess.run(cmd, shell=True, cwd=work_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("DIRAC calculation failed:")
        print(e.stderr)
        raise
    return os.path.join(work_dir, os.path.splitext(inp_file)[0] + '.out')

def parse_dirac_energy(out_file):
    with open(out_file) as f:
        text = f.read()
    m = re.search(r"Total CCSD\(T\) energy\s*:\s*([-+]?\d+\.\d+)", text)
    if not m:
        raise ValueError("CCSD(T) energy not found")
    return float(m.group(1))

def cbs_extrapolate(e_dz, e_tz):
    return (27 * e_tz - 8 * e_dz) / 19.0

def SEsolver_Rovib(coordinate, potential, mass, J):
    N = len(potential)
    L = max(coordinate) - min(coordinate)
    y = (coordinate - min(coordinate)) / L + 1e-20
    dy = abs(y[0] - y[1])
    m = mass_proton * mass
    cent = J * (J + 1) / (2 * m * (coordinate / bohr_to_Ang)**2) if J > 0 else 0.0
    d = 1 / dy**2 + m * ((L / bohr_to_Ang)**2) * (potential + cent)
    e = -1 / (2 * dy**2) * np.ones(len(d) - 1)
    w, v = eigh_tridiagonal(d, e)
    w_J = hartree_to_joule * w / (m * (L / bohr_to_Ang)**2)
    return w_J, v

def solve_symmetric_potential(R_vals, energies, mass, poly_deg=10):
    print(f"\nSolving rovibrational Schrödinger equation for rotational quantum number J = 0")
    
    poly = np.poly1d(np.polyfit(R_vals, energies, poly_deg))
    grid_R = np.linspace(R_vals.min(), R_vals.max(), 2000)
    V = poly(grid_R)
    eJ, _ = SEsolver_Rovib(grid_R, V, mass, 0)
    for v in range(min(3, len(eJ))):
        print(f"  v={v}: total electronic-rovibrational energy = {eJ[v]/hartree_to_joule:.8f} Hartree")
    omega = (eJ[1] - eJ[0]) / (planck_constant * speed_of_light) / 100.0

    
    idx_min = np.argmin(energies)
    if idx_min == 0 or idx_min == len(energies)-1:
        r_eq = R_vals[idx_min]
    else:
        x_left  = R_vals[idx_min-1]
        x_mid   = R_vals[idx_min]
        x_right = R_vals[idx_min+1]
        y_left  = energies[idx_min-1]
        y_mid   = energies[idx_min]
        y_right = energies[idx_min+1]
        denom = (x_left - x_mid) * (x_left - x_right) * (x_mid - x_right)
        if abs(denom) < 1e-14:
            r_eq = x_mid
        else:
            a = (y_left*(x_mid - x_right) + y_mid*(x_right - x_left) + y_right*(x_left - x_mid)) / denom
            b = (y_left*(x_right**2 - x_mid**2) + y_mid*(x_left**2 - x_right**2) + y_right*(x_mid**2 - x_left**2)) / denom
            if abs(a) < 1e-14:
                r_eq = x_mid
            else:
                r_eq = -b / (2*a)
            if not (x_left <= r_eq <= x_right):
                r_eq = x_mid
    # ----------------------------------------------------------------------------

    mu_kg = mass * mass_proton_SI
    B_e_J = planck_constant**2 / (8 * np.pi**2 * mu_kg * (r_eq * 1e-10)**2)
    B_e = B_e_J / (planck_constant * speed_of_light) / 100.0
    return omega, B_e, r_eq

def solve_qmode_potential(q_vals, energies, mass, poly_deg=2, rmin=-0.7, rmax=0.7):
    deg = min(poly_deg, len(q_vals)-1)
    pot = np.poly1d(np.polyfit(q_vals, energies, deg))
    grid_q = np.linspace(rmin, rmax, 1000)
    V = pot(grid_q)
    eJ, _ = SEsolver_Rovib(grid_q, V, mass, 0)
    print(f"\nSolving rovibrational Schrödinger equation for rotational quantum number J = 0")
    for v in range(min(3, len(eJ))):
        print(f"  v={v}: total electronic-rovibrational energy = {eJ[v]/hartree_to_joule:.8f} Hartree")
    omega = (eJ[1] - eJ[0]) / (planck_constant * speed_of_light) / 100.0
    return omega

def main():
    # ---------- Заголовок ----------
    header_block = """\n""" + "-"*50 + """
4cDCCCSDCBSRovibConstants by Daniil A. Shitov
Four-component Dirac-Coulomb CCSD(T)/CBS Calculation of Rovibrational Constants
Solving four-component couple cluster equations:
H = ∑ₚₐhₚₐ(aₚ)†aₐ + ½ ∑ₚₐᵣₛHₚₐᵣₛ(aₚ)†(aₐ)†aₛaᵣ
T = ∑ᵢᵅtᵢᵅ(aₐ)†aᵢ + ∑ᵢⱼᵅᵖtᵢⱼᵅᵖ(aₐ)†(aₚ)†aⱼaᵢ
|Φ(4cCCSD) ⟩ = exp[T]|Φ(DHF)⟩
E(4cCCSD) = ⟨Φ(4cCCSD)|H|Φ(4cCCSD)⟩
⟨Φ(DHF)|(aᵢ)†aₐH|Φ(4cCCSD)⟩ = 0
⟨Φ(DHF)|(aᵢ)†(aⱼ)†aₚaₐH|Φ(4cCCSD)⟩ = 0
Density matrix of four-component CCSD wavefunction:
Dₚₛ = ⟨Φ(4cCCSD)|aₚ†aₛ|Φ(4cCCSD)⟩
Solving rovibrational Schrödinger equation:
[–(ħ²/2μₖ) d²/dQₖ² + ħ² J(J+1)/(2μₖQₖ²) + E(Qₖ)]χᵥⱼ(Qₖ)Yⱼₘ(θ, φ) = Eᵥⱼχᵥⱼ(Qₖ)Yⱼₘ(θ, φ)
""" + "-"*50 + "\n"
    print(header_block)

    parser = ArgumentParser()
    parser.add_argument("xyz_file", help="Input .xyz file for YXY molecule")
    group_ham = parser.add_mutually_exclusive_group(required=True)
    group_ham.add_argument("-DC", action="store_true", help="Dirac‑Coulomb Hamiltonian")
    group_ham.add_argument("-LL", action="store_true", help="Levy‑Leblond Hamiltonian")
    parser.add_argument("-q", type=int, default=0, help="Molecular charge (integer)")
    parser.add_argument("--keep", action="store_true", help="Keep all intermediate files")
    args = parser.parse_args()

    hamiltonian = "DC" if args.DC else "LL"
    ham_name = "Dirac-Coulomb" if args.DC else "Levy-Leblond"
    wfn = "four-component CCSD(T)"
    charge = args.q

    Y, X, Z_eq = parse_xyz_triatomic(args.xyz_file)
    prefix = f"{Y}{X}{Y}_{hamiltonian}"   # для именования файлов

    print(f"Molecule: {Y}-{X}-{Y}, initial internuclear distance = {2*Z_eq:.5f} Å")
    mY = get_mass(Y)
    mX = get_mass(X)
    print(f"Nuclei masses: nucleus {Y} mass = {mY:.4f} proton mass, nucleus {X} mass = {mX:.4f} proton mass")

    bases = ['av2z', 'av3z']
    labels = ['DZ', 'TZ']

    # ---- Symmetric stretching ----
    print("\n" + "="*60)
    print("Symmetric stretching mode")
    R_min = 2*Z_eq - 0.4
    R_max = 2*Z_eq + 0.4
    R_step = 0.05
    R_vals = np.arange(R_min, R_max + R_step/2, R_step)
    print(f"Scanning internuclear distances from {R_min:.2f} to {R_max:.2f} Å, step {R_step:.2f} Å")

    E_dz = []
    E_tz = []
    for bi, basis in enumerate(bases):
        label = labels[bi]
        print(f"\n  Basis: dyall.{basis} ({label})")
        E_list = []
        for R in R_vals:
            z = R/2
            xyz_file = f"{prefix}_sym_{label}_{R:.3f}.xyz"
            generate_xyz_symmetric(Y, X, z, xyz_file)
            inp_file = f"{prefix}_sym_{label}_{R:.3f}.inp"
            write_dirac_input_triatomic(f"{Y}{X}{Y}", hamiltonian, basis, charge, inp_file)
            print(f"    Running four-component calculation with {ham_name} Hamiltonian and {wfn} wavefunction in dyall.{basis} relativistic basis set for internuclear distance = {R:.3f} Å ...")
            out_file = run_dirac(inp_file, xyz_file, ".")
            e = parse_dirac_energy(out_file)
            if not args.keep:
                for f in [xyz_file, inp_file, out_file]:
                    if os.path.exists(f): os.remove(f)
            E_list.append(e)
            print(f"      total electronic energy = {e:.8f} Hartree")
        if label == 'DZ':
            E_dz = E_list
        else:
            E_tz = E_list

    # Extrapolate to CBS for symmetric stretching
    E_cbs = [cbs_extrapolate(e_dz, e_tz) for e_dz, e_tz in zip(E_dz, E_tz)]
    cbs_sym_file = f"{prefix}_sym_CBS_potential.txt"
    np.savetxt(cbs_sym_file, np.column_stack((R_vals, E_cbs)), header="R (Å)   E_CBS (Hartree)", fmt="%.20f")
    print(f"\n  CBS potential saved to {cbs_sym_file}")

    mu_sym = mY / 2
    print(f"Effective reduced mass for symmetric stretching mode: {mu_sym:.4f} proton mass")
    omega_sym, B_e, r_eq_cbs = solve_symmetric_potential(R_vals, E_cbs, mu_sym, poly_deg=10)
    print(f"  Equilibrium internuclear distance (CBS): {r_eq_cbs:.5f} Å")

    # ---- Bending mode ----
    print("\n" + "="*60)
    print("Bending mode")
    deltas = np.arange(0.01, 0.11, 0.01)
    mu_bend = mX * (2*mY) / (mX + 2*mY)
    print(f"Effective reduced mass for bending mode: {mu_bend:.4f} proton mass")
    q_vals = []
    E_bend_dz = []
    E_bend_tz = []
    for bi, basis in enumerate(bases):
        label = labels[bi]
        print(f"\n  Basis: dyall.{basis} ({label})")
        E_list = []
        for d in deltas:
            xyz_file = f"{prefix}_bend_{label}_{d:.3f}.xyz"
            generate_xyz_bending(Y, X, Z_eq, d, mY, mX, xyz_file)
            inp_file = f"{prefix}_bend_{label}_{d:.3f}.inp"
            write_dirac_input_triatomic(f"{Y}{X}{Y}", hamiltonian, basis, charge, inp_file)
            print(f"    Running four-component calculation with {ham_name} Hamiltonian and {wfn} wavefunction in dyall.{basis} relativistic basis set for bending displacement = {d:.3f} Å ...")
            out_file = run_dirac(inp_file, xyz_file, ".")
            e = parse_dirac_energy(out_file)
            if not args.keep:
                for f in [xyz_file, inp_file, out_file]:
                    if os.path.exists(f): os.remove(f)
            q = d - (mX*d)/mY
            if bi == 0:
                q_vals.append(q)
            E_list.append(e)
            print(f"      effective bending displacement = {q:.6f} Å, total electronic energy = {e:.8f} Hartree")
        if label == 'DZ':
            E_bend_dz = E_list
        else:
            E_bend_tz = E_list

    # Extrapolate to CBS for bending
    E_bend_cbs = [cbs_extrapolate(e_dz, e_tz) for e_dz, e_tz in zip(E_bend_dz, E_bend_tz)]
    
    idx_orig = np.argmin(np.abs(R_vals - 2*Z_eq))
    E_eq_cbs = E_cbs[idx_orig]
    q_pos = np.array(q_vals)
    q_full = [0.0] + list(q_pos) + list(-q_pos)
    e_full = [E_eq_cbs] + list(E_bend_cbs) + list(E_bend_cbs)
    idx = np.argsort(q_full)
    q_full = np.array(q_full)[idx]
    e_full = np.array(e_full)[idx]
    cbs_bend_file = f"{prefix}_bend_CBS_potential.txt"
    np.savetxt(cbs_bend_file, np.column_stack((q_full, e_full)), header="q (Å)   E_CBS (Hartree)", fmt="%.20f")
    print(f"\n  CBS potential saved to {cbs_bend_file}")
    omega_bend = solve_qmode_potential(q_full, e_full, mu_bend, poly_deg=2, rmin=-0.7, rmax=0.7)

    # ---- Antisymmetric mode ----
    print("\n" + "="*60)
    print("Antisymmetric mode")
    mu_asym = mX * (2*mY) / (mX + 2*mY)
    print(f"Effective reduced mass for antisymmetric mode: {mu_asym:.4f} proton mass")
    E_asym_dz = []
    E_asym_tz = []
    for bi, basis in enumerate(bases):
        label = labels[bi]
        print(f"\n  Basis: dyall.{basis} ({label})")
        E_list = []
        for d in deltas:
            xyz_file = f"{prefix}_asym_{label}_{d:.3f}.xyz"
            generate_xyz_antisymmetric(Y, X, Z_eq, d, mY, mX, xyz_file)
            inp_file = f"{prefix}_asym_{label}_{d:.3f}.inp"
            write_dirac_input_triatomic(f"{Y}{X}{Y}", hamiltonian, basis, charge, inp_file)
            print(f"    Running four-component calculation with {ham_name} Hamiltonian and {wfn} wavefunction in dyall.{basis} relativistic basis set for antisymmetric displacement = {d:.3f} Å ...")
            out_file = run_dirac(inp_file, xyz_file, ".")
            e = parse_dirac_energy(out_file)
            if not args.keep:
                for f in [xyz_file, inp_file, out_file]:
                    if os.path.exists(f): os.remove(f)
            E_list.append(e)
            q_eff = d - (mX*d)/mY
            print(f"      effective antisymmetric displacement = {q_eff:.6f} Å, total electronic energy = {e:.8f} Hartree")
        if label == 'DZ':
            E_asym_dz = E_list
        else:
            E_asym_tz = E_list

    E_asym_cbs = [cbs_extrapolate(e_dz, e_tz) for e_dz, e_tz in zip(E_asym_dz, E_asym_tz)]
    # Энергия в нуле берётся для ИСХОДНОЙ геометрии
    idx_orig = np.argmin(np.abs(R_vals - 2*Z_eq))
    E_eq_cbs = E_cbs[idx_orig]
    q_full = [0.0] + list(q_pos) + list(-q_pos)
    e_full = [E_eq_cbs] + list(E_asym_cbs) + list(E_asym_cbs)
    idx = np.argsort(q_full)
    q_full = np.array(q_full)[idx]
    e_full = np.array(e_full)[idx]
    cbs_asym_file = f"{prefix}_asym_CBS_potential.txt"
    np.savetxt(cbs_asym_file, np.column_stack((q_full, e_full)), header="q (Å)   E_CBS (Hartree)", fmt="%.20f")
    print(f"\n  CBS potential saved to {cbs_asym_file}")
    omega_asym = solve_qmode_potential(q_full, e_full, mu_asym, poly_deg=2, rmin=-0.7, rmax=0.7)

    # ---- Output results ----
    results_block = f"""
{"="*60}
CBS extrapolation results
{"="*60}

  Hamiltonian: {ham_name}
  Wavefunction: {wfn}
  Point group of equilibrium nuclear configuration: D(∞,h)
  Equilibrium internuclear distance: {r_eq_cbs:.6f} Å
  The equilibrium value of the effective coordinate corresponding to the bending mode: 0.000000 Å
  The equilibrium value of the effective coordinate corresponding to the antisymmetric mode: 0.000000 Å
  Fundamental symmetric stretching vibrational frequency: {omega_sym:.4f} 1/cm
  Fundamental bending vibrational frequency: {omega_bend:.4f} 1/cm
  Fundamental antisymmetric vibrational frequency: {omega_asym:.4f} 1/cm
  Rotational constant with respect to the equilibrium nuclear configuration: {B_e:.6f} 1/cm
{"="*60}
"""
    print(results_block)

    # Сохраняем summary в файл
    summary_file = f"{prefix}_rovib_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(header_block)
        f.write(results_block)
    print(f"Summary written to {summary_file}")

    # Cleanup
    if not args.keep:
        for pattern in [f"{prefix}_sym_*.xyz", f"{prefix}_sym_*.inp", f"{prefix}_sym_*.out",
                        f"{prefix}_bend_*.xyz", f"{prefix}_bend_*.inp", f"{prefix}_bend_*.out",
                        f"{prefix}_asym_*.xyz", f"{prefix}_asym_*.inp", f"{prefix}_asym_*.out"]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except:
                    pass
        print("\nIntermediate files removed. Use --keep to retain all files.")

if __name__ == "__main__":
    main()
