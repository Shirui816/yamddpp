#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

from numpy import array
import numpy as np
from io import StringIO
from xml.etree import cElementTree  # pypy will be a bit slower than python
from pandas import read_csv
from System import System


class HoomdXml(System):
    @staticmethod
    def _get_attrib(dd):
        dt = eval('[' + ','.join(["('%s', int)" % key for key in dd.keys()]) + ']')
        values = [tuple(dd.values())]
        return array(values, dtype=dt)

    def __init__(self, filename, topology=False, needed=None, with_body=False):
        super(HoomdXml, self).__init__(filename=filename, topology=topology, with_body=with_body)
        tree = cElementTree.ElementTree(file=self.filename)
        root = tree.getroot()
        configuration = root[0]
        self.configure = self._get_attrib(configuration.attrib)
        self.nodes = {}
        for e in configuration:
            if e.tag == 'box':
                # print(e.attrib)
                self.box = np.array([float(e.attrib['lx']), float(e.attrib['ly']), float(e.attrib['lz'])])
                continue
            if (needed is None) or (e.tag in needed):
                continue
            self.nodes[e.tag] = read_csv(StringIO(e.text), delim_whitespace=True, squeeze=1, header=None).values
        self.position = self.nodes.get('position')
        self.type = self.nodes.get('type')
        self.velocity = self.nodes.get('velocity')
        self.image = self.nodes.get('image')
        self.bond = self.nodes.get('bond')
        self.angle = self.nodes.get('angle')
        self.dihedral = self.nodes.get('dihedral')
        self.orientation = self.nodes.get('orientation')
        self.mass = self.nodes.get('mass')
        self.diameter = self.nodes.get('diameter')
        self.body = self.nodes.get('body')
        self.natoms = self.configure['natoms'][0]
        self.timestep = self.configure['time_step'][0]
        # Momentum of Inertia
        if self.topology:
            self._get_topo()
