#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import warnings
from TopologicalAnalysis import grab_iter_dual, classify_isomers, bond_hash_dualdirect, body_hash
import numpy as np
from DataRead import HoomdXml


class System(object):
    def __init__(self, read_object, topology=False, with_body=False):  # don't read topology by default
        self.topology = topology
        self.with_body = with_body
        self.position = read_object.position
        self.type = read_object.type
        self.bond = read_object.bond
        self.diameter = read_object.diameter
        self.body = read_object.body
        self.natoms = read_object.natoms
        self.trajectory = {}

    def _get_topo(self):
        self._check_params()
        self.body_hash = body_hash(self.body) if self.with_body else None
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
        if (self.body is None) and self.with_body:
            warnings.warn("Warning, you wanted to classify molecules with body but no body data was given！")
        if self.bond is None:
            raise(ValueError("You sure you haven't got any bonds? Why would you collect molecules for monomers?"))

    def add_trajectory(self, *args, **kwargs):
        for f in args:
            if '.xml' in f:
                xml = HoomdXml(f, needed=kwargs['needed'])
                self.trajectory[xml.timestep] = xml
        if len(args) == 1:
            if '.dcd' in args[0]:
                pass
