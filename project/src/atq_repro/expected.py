from __future__ import annotations


PAPER_BENCHMARK_NO_GADGETS = {
    "8_bit_adder": 139,
    "barenco_tof_3": 13,
    "barenco_tof_4": 23,
    "barenco_tof_5": 33,
    "barenco_tof_10": 83,
    "csla_mux_3": 39,
    "csum_mux_9": 71,
    "gf_2pow2_mult": 17,
    "gf_2pow3_mult": 29,
    "gf_2pow4_mult": 39,
    "gf_2pow5_mult": 59,
    "gf_2pow6_mult": 77,
    "gf_2pow7_mult": 104,
    "gf_2pow8_mult": 123,
    "gf_2pow9_mult": 161,
    "gf_2pow10_mult": 196,
    "grover_5": 152,
    "hamming_15_high": 773,
    "hamming_15_low": 73,
    "hamming_15_med": 156,
    "hwb_6": 51,
    "mod_adder_1024": 762,
    "mod_mult_55": 17,
    "mod_red_21": 51,
    "mod_5_4": 7,
    "qcla_adder_10": 135,
    "qcla_com_7": 59,
    "qcla_mod_7": 199,
    "qft_4": 53,
    "rc_adder_6": 37,
    "nc_tof_3": 13,
    "nc_tof_4": 19,
    "nc_tof_5": 25,
    "nc_tof_10": 55,
    "vbe_adder_3": 19,
}

PAPER_BENCHMARK_GADGETS = {
    "8_bit_adder": 94,
    "barenco_tof_3": 4,
    "barenco_tof_4": 8,
    "barenco_tof_5": 12,
    "barenco_tof_10": 32,
    "csla_mux_3": 16,
    "csum_mux_9": 28,
    "gf_2pow2_mult": 6,
    "gf_2pow3_mult": 12,
    "gf_2pow4_mult": 18,
    "gf_2pow5_mult": 26,
    "gf_2pow6_mult": 36,
    "gf_2pow7_mult": 44,
    "gf_2pow8_mult": 58,
    "gf_2pow9_mult": 70,
    "gf_2pow10_mult": 92,
    "grover_5": 66,
    "hamming_15_high": 440,
    "hamming_15_low": 34,
    "hamming_15_med": 78,
    "hwb_6": 20,
    "mod_adder_1024": 500,
    "mod_mult_55": 6,
    "mod_red_21": 22,
    "mod_5_4": 2,
    "qcla_adder_10": 94,
    "qcla_com_7": 24,
    "qcla_mod_7": 122,
    "qft_4": 44,
    "rc_adder_6": 12,
    "nc_tof_3": 4,
    "nc_tof_4": 6,
    "nc_tof_5": 8,
    "nc_tof_10": 18,
    "vbe_adder_3": 6,
}


def expected_binary_addition_effective_tcount() -> dict[str, int]:
    return {f"cuccaro_adder_n{bits}": 2 * (bits - 1) for bits in range(3, 11)}


def expected_gf_gadget_effective_tcount() -> dict[str, int]:
    toffoli = {
        2: 3,
        3: 6,
        4: 9,
        5: 13,
        6: 18,
        7: 22,
        8: 29,
        9: 35,
        10: 46,
    }
    return {f"gf_2pow{m}_mult": 2 * value for m, value in toffoli.items()}


def expected_gf_no_gadget_tcount() -> dict[str, int]:
    return {
        "gf_2pow2_mult": 17,
        "gf_2pow3_mult": 29,
        "gf_2pow4_mult": 39,
        "gf_2pow5_mult": 59,
        "gf_2pow6_mult": 77,
        "gf_2pow7_mult": 104,
        "gf_2pow8_mult": 123,
        "gf_2pow9_mult": 161,
        "gf_2pow10_mult": 196,
    }
