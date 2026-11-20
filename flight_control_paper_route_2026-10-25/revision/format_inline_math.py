"""One-time, bounded conversion of this revision's prose notation to inline TeX.

Existing TeX, links, code spans and HTML tags are protected. The manuscript is
the editable source afterwards; this is a transformation record, not a build step.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'webpage/dhc6_report.md'

PHRASES = {
    'e_{k+1}=p_ref,k+1−p_k+1': r'e_{k+1}=p_{\mathrm{ref},k+1}-p_{k+1}',
    'ω_φ=R_ωω_d': r'\omega_\phi=R_\omega\omega_d',
    'ζ_φ=R_ζζ_d': r'\zeta_\phi=R_\zeta\zeta_d',
    'ẋ=Ax+BF': r'\dot{x}=Ax+BF',
    'p=Cx': r'p=Cx',
    'Cx=p₀': r'Cx=p_0',
    'ṗ=−p/T_R+bF=2 deg/s²': r'\dot{p}=-p/T_R+bF=2\,\mathrm{deg/s^2}',
    'b/(s+1/T_R)': r'b/(s+1/T_R)',
    'φ̇≈p': r'\dot{\phi}\approx p',
    'G(0)=0': r'G(0)=0',
    'λ_s≠0': r'\lambda_s\ne0',
    'λ_s>0': r'\lambda_s>0',
    'ω_d²|φ/β|_d>20': r'\omega_d^2|\phi/\beta|_d>20',
    'ln(2)/λ_s': r'\ln(2)/\lambda_s',
    'g=0.9、ε=0.01': r'g=0.9,\quad\epsilon=0.01',
    'g=1.05': r'g=1.05',
    'g=1': r'g=1',
    'a²+0.25': r'a^2+0.25',
    'r=−0.1': r'r=-0.1',
    'δ=−0.1+0.995×(−4)−(−5)=0.92': r'\delta=-0.1+0.995\times(-4)-(-5)=0.92',
    '|p|>300 deg/s': r'|p|>300\,\mathrm{deg/s}',
    'ζ_d≥0、ω_d≥0.4': r'\zeta_d\ge0,\quad\omega_d\ge0.4',
    '0.20<τ_p≤0.25 s': r'0.20<\tau_p\le0.25\,\mathrm{s}',
    'ω_d=2 rad/s': r'\omega_d=2\,\mathrm{rad/s}',
    'ζ_d=0.1': r'\zeta_d=0.1',
    'ζ_d=0.5': r'\zeta_d=0.5',
    'T_R=0.5 s': r'T_R=0.5\,\mathrm{s}',
    'p=7 deg/s': r'p=7\,\mathrm{deg/s}',
    'b=8 deg/(s²·N)': r'b=8\,\mathrm{deg/(s^2\!\cdot N)}',
    'F=2 N': r'F=2\,\mathrm{N}',
    'p≈7.04 deg/s': r'p\approx7.04\,\mathrm{deg/s}',
    'a=0.1': r'a=0.1',
    'a=0': r'a=0',
    'd=1': r'd=1',
    'd=0': r'd=0',
    'log σ=−2': r'\log\sigma=-2',
    '3×10⁻⁴': r'3\times10^{-4}',
    'Kp=45、Ki=24、Kd=1': r'K_p=45,\quad K_i=24,\quad K_d=1',
    'V_v(ξ)': r'V_v(\xi)',
}
SYMBOLS = {
    'F_applied': r'F_{\mathrm{applied}}', 'F_req': r'F_{\mathrm{req}}',
    'p_ref': r'p_{\mathrm{ref}}', 'p_cmd': r'p_{\mathrm{cmd}}',
    'p₀': r'p_0', 'L_Fa': r'L_{F_a}', 'λ_s': r'\lambda_s', 'T_R': r'T_R',
    'ζ_dω_d': r'\zeta_d\omega_d', 'ζ_d': r'\zeta_d', 'ω_d': r'\omega_d',
    'R_ω': r'R_\omega', 'R_ζ': r'R_\zeta', 'τ_p': r'\tau_p',
    'ω_φ': r'\omega_\phi', 'ζ_φ': r'\zeta_\phi', 'Â': r'\widehat{A}',
    'θ': r'\theta', 'ξ': r'\xi', 'δ': r'\delta', 'φ': r'\phi',
    'β': r'\beta', 'γ': r'\gamma', 'λ': r'\lambda', 'ε': r'\epsilon',
    'w': 'w', 'p': 'p', 'r': 'r', 'x': 'x', 'e': 'e', 'a': 'a', 'u': 'u',
    'o': 'o', 'z': 'z', 'k': 'k', 'j': 'j', 'v': 'v', 'g': 'g', 'R': 'R',
}
PROTECTED = re.compile(r'(\$\$.*?\$\$|\\\(.*?\\\)|`[^`]*`|<[^>]*>|\]\([^)]*\))', re.S)


def convert_plain(text):
    phrases = re.compile('|'.join(re.escape(key) for key in sorted(PHRASES, key=len, reverse=True)))
    text = phrases.sub(lambda m: r'\(' + PHRASES[m.group()] + r'\)', text)
    parts = PROTECTED.split(text)
    pattern = re.compile(r'(?<![A-Za-z0-9_\\])(' + '|'.join(
        re.escape(key) for key in sorted(SYMBOLS, key=len, reverse=True)) + r')(?![A-Za-z0-9_])')
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(lambda m: r'\(' + SYMBOLS[m.group()] + r'\)', parts[i])
    return ''.join(parts)


def main():
    text = SOURCE.read_text()
    # A long parameter vector is broken into separately rendered symbols so
    # mobile prose can wrap between items without breaking subscripts.
    text = text.replace('参数 θ=(L_Fa, λ_s, T_R, ζ_d, ω_d, R_ω, R_ζ, τ_p)，',
                        '参数向量 θ 包含 L_Fa、λ_s、T_R、ζ_d、ω_d、R_ω、R_ζ、τ_p，')
    body, refs = text.split('## 参考文献', 1)
    parts = PROTECTED.split(body)
    for i in range(0, len(parts), 2):
        parts[i] = convert_plain(parts[i])
    text = ''.join(parts) + '## 参考文献' + refs
    # Consistent nested index and role superscripts in displayed equations.
    text = text.replace(r'L_{Fa}', r'L_{F_a}')
    text = text.replace(r'o_k^T', r'o_k^{\mathrm T}').replace(r'o_k^S', r'o_k^{\mathrm S}')
    text = text.replace(r'a^T', r'a^{\mathrm T}').replace(r'a^S_w', r'a^{\mathrm S}_w')
    SOURCE.write_text(text)
    print('Formatted prose and table notation; original display math and references preserved.')


if __name__ == '__main__':
    main()
