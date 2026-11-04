"""Relative, recovery-driven compression rounds (no absolute iteration triggers)."""
import math
import numpy as np


def survival_quotas(metadata, config):
    """Freeze declared survival budgets from the initial, reliable instance set."""
    ids = np.asarray(metadata['instance_ids'])
    fragility = np.asarray(metadata['instance_fragility'])
    boundary = np.asarray(metadata['boundary_mask'], dtype=bool)
    instance, boundaries, rows = {}, {}, []
    a, b = config['instance_protection'], config['boundary_protection']
    for k in np.unique(ids):
        k = int(k)
        mask = ids == k
        n, nb = int(mask.sum()), int(boundary[mask].sum())
        f = float(np.median(fragility[mask]))
        rho = float(np.clip(a['base_keep_ratio'] + a['fragility_beta'] * f,
                            a['base_keep_ratio'], a['max_keep_ratio']))
        floor = min(n, max(int(a['min_instance_gaussians']), int(math.ceil(rho*n)))) if k >= 0 and a['enabled'] else 0
        bf = int(math.ceil(b['boundary_keep_ratio']*nb)) if b['enabled'] else 0
        if k >= 0:
            instance[k] = floor
        boundaries[k] = bf
        rows.append(dict(instance_id=k, initial=n, initial_boundary=nb,
                         fragility=f, keep_ratio=rho, minimum=floor, boundary_minimum=bf))
    return instance, boundaries, rows


def audit_quotas(metadata, instance_min, boundary_min):
    ids = np.asarray(metadata['instance_ids'])
    counts = dict(zip(*np.unique(ids, return_counts=True)))
    boundaries = dict(zip(*np.unique(ids[np.asarray(metadata['boundary_mask'], bool)], return_counts=True)))
    violations = []
    for label, expected, observed in [('instance',instance_min,counts), ('boundary',boundary_min,boundaries)]:
        for k, floor in expected.items():
            if observed.get(k,0) < floor:
                violations.append(dict(kind=label, instance_id=int(k), expected=int(floor), actual=int(observed.get(k,0))))
    if violations:
        raise RuntimeError('Protection quota violation: '+str(violations[:10]))
    return dict(instance_violations=0, boundary_violations=0,
                instances=[dict(instance_id=int(k),current=int(counts.get(k,0)),minimum=int(v),
                                boundary_current=int(boundaries.get(k,0)),boundary_minimum=int(boundary_min.get(k,0)))
                           for k,v in instance_min.items()])


class AdaptiveSchedule:
    def __init__(self, config):
        self.config = dict(config)
        self.ratio = float(config['initial_prune_ratio'])
        self.history = []
        self.round = 0

    def round_target(self, current, final):
        return max(int(final), int(current)-max(1,int(math.ceil(current*self.ratio))))

    def recovery_done(self, history, final=False):
        c = self.config
        minimum = c.get('final_recovery_steps', c['min_recovery_steps']) if final else c['min_recovery_steps']
        maximum = max(minimum, c['max_recovery_steps']) if final else c['max_recovery_steps']
        if len(history) >= maximum:
            return True, 'maximum_recovery_steps'
        if len(history) < minimum:
            return False, 'minimum_recovery_steps'
        if not c.get('enabled',True):
            return True, 'ablation_fixed_relative_recovery'
        w = c['convergence_window']
        if len(history) >= 2*w:
            old, new = float(np.mean(history[-2*w:-w])), float(np.mean(history[-w:]))
            if abs(old-new)/max(abs(old),1e-8) < c['convergence_threshold']:
                return True, 'sliding_window_convergence'
        return False, 'recovering'

    def update(self, before, damaged, recovered):
        delta = damaged-before
        recoverability = (damaged-recovered)/(delta+1e-8) if delta > 1e-8 else float(recovered<=before)
        old = self.ratio
        if self.config.get('enabled',True):
            if recoverability > self.config.get('high_threshold',.8):
                self.ratio *= 1.1
            elif recoverability < self.config.get('low_threshold',.5):
                self.ratio *= .8
        self.ratio = float(np.clip(self.ratio,self.config['min_prune_ratio'],self.config['max_prune_ratio']))
        self.round += 1
        row = dict(round_id=self.round, before=float(before),damaged=float(damaged),recovered=float(recovered),
                   recoverability=float(recoverability), previous_ratio=old,next_ratio=self.ratio)
        self.history.append(row)
        return row
