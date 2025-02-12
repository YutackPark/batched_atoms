# type: ignore
import sys
import time
from copy import deepcopy
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import torch
import ase.io
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS

from matscipy.neighbours import neighbour_list
import sevenn._keys as KEY


"""
Author: YutackPark
Date: 2025-02-12

IMPORTANT
1. set, "export OMP_NUM_THREADS=1" to prevent unexpected parallelism
2. Tune 'nproc' for optimal performance

You have two sections to modify 1) below, 2) inside __main__
"relax" function itself is also customizable (logging each atoms, run MD, etc)
Currently, results are written as "Result.extxyz". Go to last to check.

The number of process to use is given as argument. It is also same as 'batch_size'
of GPU to run. You can use value higher than the number of cores.

No SevenNet code patch is necessary. Use SevenNetp version, or wait, or PR

TODO:
    1) Make cleaner interface
    2) MD batched example
    3) Remove unnecessary import in worker process. (may need to patch SevenNet)
"""


FMAX = 0.05
MAX_STEP = 500
use_tqdm = True


def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    ################################# CHECK BELOW ################################
    ase_filter = "frechet"
    ase_optimizer = "FIRE"
    filter_cls = {"frechet": FrechetCellFilter, "exp": ExpCellFilter}[ase_filter]
    optim_cls = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]

    nproc = int(sys.argv[1])

    num_tasks = 9
    atoms_list_src = ase.io.read("test.extxyz", index=":")
    atoms_list = deepcopy(atoms_list_src[:num_tasks])

    cutoff = 5.0  # should be determined outside
    calc_kwargs = {"model": "7net-0"}
    ################################# CHECK ABOVE ################################

    with mp.Manager() as manager:
        atoms_q = manager.Queue()
        done_q = manager.Queue()
        for id, atoms in enumerate(atoms_list):
            # atoms.info.update({"_id": id})
            atoms_q.put(atoms)

        procs = []
        pipes = []
        for id in range(nproc):
            pipe_a, pipe_b = mp.Pipe()
            procs.append(
                mp.Process(
                    target=relax,
                    args=(
                        id,
                        atoms_q,
                        done_q,
                        pipe_a,
                        cutoff,
                        filter_cls,
                        optim_cls,
                    ),
                )
            )
            pipes.append(pipe_b)
        master = mp.Process(target=calc_proc, args=(pipes,), kwargs=calc_kwargs)

        end = time.time()
        for p in procs:
            p.start()
        master.start()

        results = []
        completed_tasks = 0
        with tqdm(total=num_tasks, disable=(not use_tqdm)) as pbar:
            while completed_tasks < num_tasks:
                results.append(done_q.get())  # Get a completed job
                completed_tasks += 1
                pbar.update(1)  # Update progress bar

        for p in procs:
            p.join()
        master.join()

        workingtime = time.time() - end
        print(f"Working time: {workingtime}")
        print(len(results))
        ase.io.write("Result.extxyz", results)


def _correct_scalar(v):
    if isinstance(v, np.ndarray):
        v = v.squeeze()
        assert v.ndim == 0, f'given {v} is not a scalar'
        return v
    elif isinstance(v, (int, float, np.integer, np.floating)):
        return np.array(v)
    else:
        assert False, f'{type(v)} is not expected'


def _graph_build_matscipy(cutoff: float, pbc, cell, pos):
    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(pos)) + 1

    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]
    # it does not have self-interaction
    edge_src, edge_dst, edge_vec, shifts = neighbour_list(
        quantities='ijDS',
        pbc=pbc,
        cell=cell,
        positions=pos,
        cutoff=cutoff,
    )
    # dtype issue
    edge_src = edge_src.astype(np.int64)
    edge_dst = edge_dst.astype(np.int64)

    return edge_src, edge_dst, edge_vec, shifts

_graph_build_f = _graph_build_matscipy

def unlabeled_atoms_to_graph(atoms: ase.Atoms, cutoff: float):
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    pbc = atoms.get_pbc()

    edge_src, edge_dst, edge_vec, shifts = _graph_build_f(cutoff, pbc, cell, pos)

    edge_idx = np.array([edge_src, edge_dst])

    atomic_numbers = atoms.get_atomic_numbers()

    cell = np.array(cell)
    vol = _correct_scalar(atoms.cell.volume)
    if vol == 0:
        vol = np.array(np.finfo(float).eps)

    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: shifts,
        KEY.CELL_VOLUME: vol,
        KEY.NUM_ATOMS: _correct_scalar(len(atomic_numbers)),
    }
    data[KEY.INFO] = {}
    return data


class BatchCallCalculator(Calculator):
    def __init__(self, id, cutoff, pipe):
        super().__init__()
        self.id = id
        self.cutoff = cutoff
        self.pipe = pipe
        self.implemented_properties = [
            "free_energy",
            "energy",
            "forces",
            "stress",
            "energies",
        ]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        graph = unlabeled_atoms_to_graph(atoms.copy(), self.cutoff)
        self.pipe.send((graph))
        results = self.pipe.recv()
        self.results = results


def relax(id, atoms_q, done_q, pipe, cutoff, filter_cls, optim_cls):
    calc = BatchCallCalculator(id, cutoff, pipe)
    while True:
        if not atoms_q.empty():
            atoms = atoms_q.get(timeout=10)
        else:
            pipe.send(False)
            break
        # atoms_id = atoms.info.pop("_id")
        atoms.calc = calc
        atoms = filter_cls(atoms)
        optim = optim_cls(atoms, logfile="/dev/null")
        optim.run(FMAX, MAX_STEP)
        done_q.put(atoms.atoms.copy())  # TODO: I hate Filter impl
        del optim
        del atoms


def calc_proc(pipe_list, **calc_kwargs):
    from torch_geometric.loader.dataloader import Collater
    from sevenn.calculator import SevenNetCalculator
    from sevenn.atom_graph_data import AtomGraphData
    import sevenn.util as util

    collater = Collater([])
    calc = SevenNetCalculator(**calc_kwargs)
    model = calc.model
    model.set_is_batch_data(True)
    num_workers = len(pipe_list)

    done_flags = [False] * num_workers
    while True:
        graph_list = []
        for ii, pipe in enumerate(pipe_list):
            if done_flags[ii]:
                continue
            try:
                graph = pipe.recv()
            except Exception as e:
                graph = False
            if graph is not False:
                graph_torch = AtomGraphData.from_numpy_dict(graph)
                graph_torch.pipe_idx = ii
                graph_list.append(graph_torch)
            else:
                done_flags[ii] = True

        if len(graph_list) == 0:
            assert all(done_flags)
            break

        graph_batch = collater(graph_list)
        graph_batch.to(calc.device)
        output = model(graph_batch)
        output_list = util.to_atom_graph_list(output)

        for out in output_list:
            out.inferred_stress = out.inferred_stress.reshape(6)
            results = calc.output_to_results(out)  # fix in sevennet, to static func
            pipe_idx = out.pipe_idx
            pipe = pipe_list[pipe_idx]
            pipe.send(results)


if __name__ == "__main__":
    main()
