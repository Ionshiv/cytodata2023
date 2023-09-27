import torch as tch
from torch_geometric.data import Data#, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import os

class GraphMake():
    def __init__(self, path_data, y_name, sep=','):
        self.path_data = path_data
        self.y_name = y_name
        self.sep=sep
        if not os.path.exists(f'{path_data}.pt'):
            print('Graph from Smiles not found. Generating... this may take a while')
            df = self.read_smiles_data()
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            data_pyg = df.apply(self.make_pyg, axis=1)
            data_pyg = data_pyg[data_pyg.apply(lambda x: len(x.edge_index.shape) != 1)]
            data_pyg.reset_index(drop=True, inplace=True)
            tch.save(data_pyg, f'{path_data}.pt')
        else:
            print('Graph from Smiles found! loading...')
            data_pyg = tch.load(f'{path_data}.pt')
        self.GraphFrame = data_pyg



    def getPyG(self):
        return self.GraphFrame

    def smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return list(fp.ToBitString())

    def smiles_to_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # if mol is not None:
        #     mol = Chem.AddHs(mol)
        return mol

    def read_smiles_data(self):
        df = pd.read_csv(self.path_data, sep=self.sep)
        df['fingerprint'] = df['SMILES'].apply(self.smiles_to_fingerprint)
        df['fingerprint'] = df['fingerprint'].apply(lambda x: [int(bit) for bit in x])
        df['fingerprint'] = df['fingerprint'].apply(lambda x: np.array(x))
        df['mol'] = df['SMILES'].apply(self.smiles_to_mol)
        return df

    def is_hydrogen_donor(self, atomic_num, hybridization):
        return int((atomic_num == 8 or atomic_num == 7) and (hybridization == 3 or hybridization == 2))

    def is_polar_bond(self, atom1_num, atom2_num, electronegativity):
        en1 = electronegativity.get(atom1_num, None)
        en2 = electronegativity.get(atom2_num, None)
        if en1 is None or en2 is None:
            return 0  # Unknown electronegativity, consider as non-polar
        return int(abs(en1 - en2) > 0.4)

    def electroneg(self):
        return {
        1: 2.20,  # H
        3: 0.98,  # Li
        4: 1.57,  # Be
        5: 2.04,  # B
        6: 2.55,  # C
        7: 3.04,  # N
        8: 3.44,  # O
        9: 3.98,  # F
        11: 0.93, # Na
        12: 1.31, # Mg
        13: 1.61, # Al
        14: 1.90, # Si
        15: 2.19, # P
        16: 2.58, # S
        17: 3.16, # Cl
        19: 0.82, # K
        20: 1.00, # Ca
        22: 1.54, # Ti
        24: 1.66, # Cr
        25: 1.55, # Mn
        26: 1.83, # Fe
        27: 1.88, # Co
        28: 1.91, # Ni
        29: 1.90, # Cu
        30: 1.65, # Zn
        35: 2.96, # Br
        53: 2.66, # I
    }


    def make_pyg(self, row):
        # Create node features
        mol = row['mol']
        # pauling = electroneg()
        atom_num = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atom_hyb = [int(atom.GetHybridization()) for atom in mol.GetAtoms()]
        atom_deg = [atom.GetDegree() for atom in mol.GetAtoms()]
        atom_arom = [int(atom.GetIsAromatic()) for atom in mol.GetAtoms()]  # Aromaticity
        atom_hydrogens = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]  # Number of hydrogens
        # atom_h_donor = [is_hydrogen_donor(num, hyb) for num, hyb in zip(atom_num, atom_hyb)]
        atom_charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]  # Formal charge
        atom_chiral_tag = [int(atom.GetChiralTag()) for atom in mol.GetAtoms()]  # Chirality
        atom_val = [atom.GetExplicitValence() for atom in mol.GetAtoms()]
        #atom_mass = [atom.GetMass() for atom in mol.GetAtoms()]
        #atom_pauling = [pauling.get(num, 0) for num in atom_num]
        
        x1 = tch.tensor(atom_num, dtype=tch.float).view(-1, 1)
        x2 = tch.tensor(atom_hyb, dtype=tch.float).view(-1, 1)
        x3 = tch.tensor(atom_deg, dtype=tch.float).view(-1, 1)
        x4 = tch.tensor(atom_arom, dtype=tch.float).view(-1, 1)
        x5 = tch.tensor(atom_hydrogens, dtype=tch.float).view(-1, 1)
        x6 = tch.tensor(atom_charge, dtype=tch.float).view(-1, 1)
        x7 = tch.tensor(atom_chiral_tag, dtype=tch.float).view(-1, 1)
        x8 = tch.tensor(atom_val, dtype=tch.float).view(-1, 1)
        # x9 = tch.tensor(atom_h_donor, dtype=tch.float).view(-1, 1)
        #x10 = tch.tensor(atom_mass, dtype=tch.float).view(-1, 1)
        #x11 = tch.tensor(atom_pauling, dtype=tch.float).view(-1, 1)
        
        y = tch.tensor(row[str(self.y_name)], dtype=tch.float).view(-1, 1)
        x = tch.cat([x1
                    , x2
                    , x3
                    , x4
                    , x5
                    , x6
                    , x7
                    , x8
                    # , x9
                    # , x10
                    #, x11
                    ], dim=1)
        
        # Create edge features (connectivity)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append((i, j))
            bond_type = bond.GetBondTypeAsDouble()
            is_conjugated = int(bond.GetIsConjugated())  # Conjugation
            is_in_ring = int(bond.IsInRing())  # Ring membership
            bond_stereo = int(bond.GetStereo())  # Stereo configuration
            #bond_polarity = is_polar_bond(atom_num[i], atom_num[j], pauling)

            edge_features.append([bond_type
                                , is_conjugated
                                , is_in_ring
                                , bond_stereo
                                #, bond_polarity
                                ])
        
        edge_index = tch.tensor(edge_indices, dtype=tch.long).t().contiguous()
        edge_attr = tch.tensor(edge_features, dtype=tch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        data.smiles = row['SMILES']
        
        return data