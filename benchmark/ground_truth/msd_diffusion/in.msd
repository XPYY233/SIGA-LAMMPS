# Mean-square-displacement diffusion — reference implementation.
#
# Two details decide whether this measures anything at all:
#   * the temperature is 1.5, above the LJ melting point, so the system is
#     genuinely fluid. In a solid the MSD plateaus and D is meaningless.
#   * `com yes` removes centre-of-mass drift, which would otherwise be counted
#     as diffusion.

units           lj
atom_style      atomic
boundary        p p p

lattice         fcc 0.7
region          box block 0 6 0 6 0 6
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

velocity        all create 1.5 12345 loop geom
neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no

fix             1 all nvt temp 1.5 1.5 0.5
timestep        0.005

compute         msd all msd com yes
fix             2 all ave/time 100 1 100 c_msd[4] file msd.dat

thermo          100
thermo_style    custom step temp c_msd[4]
run             10000
