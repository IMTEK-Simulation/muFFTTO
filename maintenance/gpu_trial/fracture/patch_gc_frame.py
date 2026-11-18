"""Fracture toughness as a field: the beams keep Gc, the frame gets Gc_frame_factor * Gc.

usage: python patch_gc_frame.py IN.py OUT.py
"""
import sys

s = open(sys.argv[1]).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("""Gc = 0.1                                   # fracture toughness
""", """Gc = 0.1                                   # fracture toughness (of the beams)
# the frame gets Gc_frame_factor * Gc, so that only the beams (sticks) can crack
Gc_frame_factor = 10.0
""")
rep("from muFFTTO import io_utils  # noqa: E402\n",
    "from muFFTTO import io_utils  # noqa: E402\nfrom muFFTTO import geometry as geo  # noqa: E402\n")
rep("""mu_solid = E_solid / (2 * (1 + nu_solid))
""", """# fracture toughness at the quadrature points: Gc in the beams, Gc_frame_factor * Gc elsewhere
# (beams as in contact_test_geometry_2 / contact_beam_root_notch, from the frame wall to the tip)
beams = geo.box([0.15, 0.35], [0.52, 0.45]) | geo.box([0.47, 0.55], [0.85, 0.65])
beam_pixel = geo.indicator(beams, np.asarray(disc_u.fft.coords)) * (phase_pixel > 0)
Gc_q = xp.broadcast_to(xp.asarray(np.where(beam_pixel > 0, Gc, Gc_frame_factor * Gc)),
                       (nb_q, *phase_pixel.shape))                                # [q, x, y]

mu_solid = E_solid / (2 * (1 + nu_solid))
""")
rep('''def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, weight_q=1.0):
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= Gc * length_scale * weight_q''',
'''def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, weight_q=1.0, gc_q=Gc):
    """gc_q: fracture toughness, a number (preconditioner) or per quadrature point (Gc_q)."""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= gc_q * length_scale * weight_q''')
rep("""    apply_damage_operator(x, Ax, reaction.s, damage_weight_q)""",
    """    apply_damage_operator(x, Ax, reaction.s, damage_weight_q, gc_q=Gc_q)""")
rep("""    reaction.s[...] = Gc / length_scale + 2.0 * history.s""",
    """    reaction.s[...] = Gc_q / length_scale + 2.0 * history.s""")
rep("""                  'length_scale': length_scale,""",
    """                  'length_scale': length_scale, 'Gc_frame_factor': Gc_frame_factor,""")
rep("""                f'_N{nnn}_Gc{Gc:g}_prec-{mechanics_preconditioner}'""",
    """                f'_N{nnn}_Gc{Gc:g}_Gcframe{Gc_frame_factor:g}x_prec-{mechanics_preconditioner}'""")
open(sys.argv[2], "w").write(s)
print("ok")
