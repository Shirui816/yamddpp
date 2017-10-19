#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

from abc import ABCMeta  # , abstractmethod
import warnings
from TopologicalAnalysis import grab_iter_dual, classify_isomers, bond_hash_dualdirect, body_hash
import numpy as np


class System(metaclass=ABCMeta):
    def __init__(self, filename, topology=False, with_body=False, is_traj=False):  # don't read topology by default
        self.filename = filename
        self.topology = topology
        self.withbody = with_body
        if is_traj:
           self.trajectory = None
        self.position = None
        self.type = None
        self.velocity = None
        self.image = None
        self.bond = None
        self.angle = None
        self.dihedral = None
        self.orientation = None
        self.mass = None
        self.diameter = None
        self.body = None
        self.natoms = None

    def _get_topo(self):
        self._check_params()
        self.body_hash = body_hash(self.body) if self.withbody else None
        self.bond_hash = bond_hash_dualdirect(self.bond, self.natoms)
        print("Catching Molecules...")
        molecular_hash = {}
        molecular_list = []
        for i in range(self.natoms):
            molecular_hash[i] = False
        for i in range(self.natoms):
            # if molecular_hash[i] == True:
                # continue
            # gb(i, self.body_hash, self.bond_hash_nn, mol_idxes=molecular_idxs,mol_used=molecular_hash)
            molecular_idxs = grab_iter_dual(i, self.bond_hash, mol_used=molecular_hash, body_hash=self.body_hash)
            # self.body_hash)
            if not len(molecular_idxs) <= 1:  # avoid monomer and void list
                molecular_list.append(molecular_idxs)
            for atom in molecular_idxs:
                molecular_hash[atom] = True
        # while [] in molecular_list: # this remove operation is really SLOW
            # molecular_list.remove([])
        self.molecules = np.array([np.array(x) for x in molecular_list])
        # print(self.mol_idxes)
        molecular_types = []
        # molecular_bodies = []
        for m in self.molecules:
            molecular_types.append(self.type[np.array(m)])
            # molecular_bodies.append(self.nodes['body'][np.array(m)])
        print("Done.")
        self.molecules_type = np.array([np.array(x) for x in molecular_types])
        self.isomers = {}
        print("Classifying isomers...")
        classify_isomers(self.molecules_type, self.isomers)
        print("Done.")
        return self

    def _check_params(self):
        if self.type is None:
            raise(ValueError("No type found, I can't gathering topological information!"))
        if (self.body is None) and self.withbody:
            warnings.warn("Warning, you wanted to classify molecules with body but no body data was given！")
        if self.bond is None:
            raise(ValueError("You sure you haven't got any bonds? Why would you collect molecules for monomers?"))
