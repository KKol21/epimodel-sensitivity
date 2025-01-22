from . import MockModelBase


class MockContactModel(MockModelBase):
    def __init__(self, data):
        super().__init__(data)

    def odefun(self, t, y):
        (s, l1, l2, ip, ia1, ia2, ia3, is1, is2, is3, h, ic, icr, r, d) = self.get_comp_vals(y)
        ps = self.ps

        # Compute transmission
        infectious_terms = ip + ps["inf_a"] * (ia1 + ia2 + ia3) + (is1 + is2 + is3)
        transmission = self.get_transmission(infectious_terms) * ps["susc"]

        ds = -s * transmission
        dl1 = s * transmission - 2 * ps["alpha_l"] * l1
        dl2 = 2 * ps["alpha_l"] * l1 - 2 * ps["alpha_l"] * l2
        dip = 2 * ps["alpha_l"] * l2 - ps["alpha_p"] * ip

        dia1 = ps["p"] * ps["alpha_p"] * ip - 3 * ps["gamma_a"] * ia1
        dia2 = 3 * ps["gamma_a"] * ia1 - 3 * ps["gamma_a"] * ia2
        dia3 = 3 * ps["gamma_a"] * ia2 - 3 * ps["gamma_a"] * ia3

        dis1 = (1 - ps["p"]) * ps["alpha_p"] * ip - 3 * ps["gamma_s"] * is1
        dis2 = 3 * ps["gamma_s"] * is1 - 3 * ps["gamma_s"] * is2
        dis3 = 3 * ps["gamma_s"] * is2 - 3 * ps["gamma_s"] * is3

        dh = ps["h"] * (1 - ps["xi"]) * 3 * ps["gamma_s"] * is3 - ps["gamma_h"] * h
        dic = ps["h"] * ps["xi"] * 3 * ps["gamma_s"] * is3 - ps["gamma_c"] * ic
        dicr = (1 - ps["mu"]) * ps["gamma_c"] * ic - ps["gamma_cr"] * icr

        dr = (
            3 * ps["gamma_a"] * ia3
            + (1 - ps["h"]) * 3 * ps["gamma_s"] * is3
            + ps["gamma_h"] * h
            + ps["gamma_cr"] * icr
        )
        dd = ps["mu"] * ps["gamma_c"] * ic

        return self.concat_sol(
            ds, dl1, dl2, dip, dia1, dia2, dia3, dis1, dis2, dis3, dh, dic, dicr, dr, dd
        )
