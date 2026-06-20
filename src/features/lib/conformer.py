"""Conformer generation: ETKDGv3 (useSmallRingTorsions=True) + MMFF94s x50, lowest-E (first-by-index tie-break).

MMFF return codes: 0 = converged, 1 = not fully minimized in maxIters, -1 = optimization failed.
We accept rc in {0, 1} and stop on rc == -1.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

N_CONFORMERS = 50
MMFF_MAX_ITERS = 2000
SEED = 42


def generate_lowest_e_conformer(mol: Chem.Mol) -> Chem.Mol:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=N_CONFORMERS, params=params))
    assert len(cids) > 0, "conformer: ETKDGv3 embedding failed (0 confs)"
    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol_h, maxIters=MMFF_MAX_ITERS, mmffVariant="MMFF94s"
    )
    assert len(results) == len(cids), \
        f"conformer: MMFF result count {len(results)} != cid count {len(cids)}"
    rcs = [rc for rc, _ in results]
    energies = [e for _, e in results]
    for cid, rc in zip(cids, rcs):
        assert rc != -1, f"conformer: MMFF94s optimization failed (cid={cid}, rc=-1)"
    best_idx = energies.index(min(energies))  # first-by-index tie-break
    best_cid = cids[best_idx]
    result = Chem.Mol(mol_h)
    result.RemoveAllConformers()
    result.AddConformer(mol_h.GetConformer(best_cid), assignId=True)
    return result
