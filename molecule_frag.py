from rdkit import Chem
from rdkit.Chem import BRICS
import random
import re



# Input SMILES string
smiles = 'CC(=O)Nc1ccc(O)cc1'
mol = Chem.MolFromSmiles(smiles)

# Find BRICS bonds to break
brics_bonds = BRICS.FindBRICSBonds(mol)
brics_bonds = [x[0] for x in brics_bonds]

# Record BRICS bond information and assign a unique dummy atom label to each broken bond
broken_bonds = []
bond_indices_set = set()  # Used to track already added bond indices
for idx, (atom_idx1, atom_idx2) in enumerate(brics_bonds):
    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    bond_idx = bond.GetIdx()
    bond_type = bond.GetBondType()
    # Use isotopic labels to differentiate each broken bond
    dummy_label = idx + 1
    broken_bonds.append({
        'bond_idx': bond_idx,
        'atom_idx1': atom_idx1,
        'atom_idx2': atom_idx2,
        'bond_type': bond_type,
        'dummy_label': (dummy_label, dummy_label)
    })
    bond_indices_set.add(bond_idx)

# After BRICS bonds, find bonds between rings and side chains and add to the broken bond list
# First, get all bonds
current_dummy_label = len(broken_bonds) + 1  # Update dummy atom label

for bond in mol.GetBonds():
    bond_idx = bond.GetIdx()
    # Check if bond is already in broken_bonds
    if bond_idx in bond_indices_set:
        continue  # Skip already added bonds

    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    bond_type = bond.GetBondType()

    # Check if bond connects a ring and a side chain
    atom1_in_ring = atom1.IsInRing()
    atom2_in_ring = atom2.IsInRing()

    if atom1_in_ring != atom2_in_ring:
        # This bond connects a ring and a side chain
        dummy_label = current_dummy_label
        current_dummy_label += 1
        broken_bonds.append({
            'bond_idx': bond_idx,
            'atom_idx1': atom1.GetIdx(),
            'atom_idx2': atom2.GetIdx(),
            'bond_type': bond_type,
            'dummy_label': (dummy_label, dummy_label)
        })
        bond_indices_set.add(bond_idx)  # Add to set

# Get bond indices and dummy labels for all bonds to be broken
bond_indices = [bond['bond_idx'] for bond in broken_bonds]
dummy_labels = [bond['dummy_label'] for bond in broken_bonds]

# Break the molecule at the specified bonds and add dummy atoms with labels
mol_frag = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True, dummyLabels=dummy_labels)

print_fragments = Chem.GetMolFrags(mol_frag, asMols=True, sanitizeFrags=False)

for i, frag in enumerate(print_fragments):
    smiles_frag = Chem.MolToSmiles(frag)
    print(f"Fragment {i + 1}: {smiles_frag}")

# Get the mapping of each atom to its fragment
frags = Chem.GetMolFrags(mol_frag, asMols=False, sanitizeFrags=False)
atom_idx_to_frag_id = {}
for frag_id, atom_indices in enumerate(frags):
    for atom_idx in atom_indices:
        atom_idx_to_frag_id[atom_idx] = frag_id

# Randomly select a fragment to discard
frag_ids = set(atom_idx_to_frag_id.values())
discarded_frag_id = random.choice(list(frag_ids))
print(f"Discarded fragment ID: {discarded_frag_id}", f"Total number of fragments: {len(list(frag_ids))}")

# Get the atom indices to discard
atoms_to_discard = [atom_idx for atom_idx, frag_id in atom_idx_to_frag_id.items() if frag_id == discarded_frag_id]

# Create an editable molecule object
mol_edit = Chem.RWMol(mol_frag)

# Remove atoms to discard in descending order to avoid indexing issues
atoms_to_discard.sort(reverse=True)
for atom_idx in atoms_to_discard:
    mol_edit.RemoveAtom(atom_idx)

# Update the atom index mapping from old indices to new indices
old_to_new_atom_idx = {}
new_idx = 0
for old_idx in range(mol_frag.GetNumAtoms()):
    if old_idx not in atoms_to_discard:
        old_to_new_atom_idx[old_idx] = new_idx
        new_idx += 1

# Create a mapping of dummy labels to dummy atom indices
dummy_label_to_dummy_atoms = {}
for atom in mol_edit.GetAtoms():
    if atom.GetAtomicNum() == 0:  # Dummy atom
        isotope = atom.GetIsotope()
        if isotope not in dummy_label_to_dummy_atoms:
            dummy_label_to_dummy_atoms[isotope] = []
        dummy_label_to_dummy_atoms[isotope].append(atom.GetIdx())

# Create an editable molecule object for reconstruction
mol_reconstructed = mol_edit

# Prepare a list to store reconnection operations
reconnection_ops = []

# Collect all reconnection operation information
for broken_bond in broken_bonds:
    dummy_label = broken_bond['dummy_label'][0]
    bond_type = broken_bond['bond_type']

    # Check if both dummy atoms exist
    if dummy_label in dummy_label_to_dummy_atoms:
        dummy_atom_indices = dummy_label_to_dummy_atoms[dummy_label]
        if len(dummy_atom_indices) != 2:
            # Cannot reconnect if the number of dummy atoms is not 2
            continue

        idx1, idx2 = dummy_atom_indices

        # Get the neighbors of the dummy atoms (actual atoms)
        atom1 = mol_reconstructed.GetAtomWithIdx(idx1)
        atom2 = mol_reconstructed.GetAtomWithIdx(idx2)

        neighbor1 = [n for n in atom1.GetNeighbors() if n.GetAtomicNum() != 0]
        neighbor2 = [n for n in atom2.GetNeighbors() if n.GetAtomicNum() != 0]

        if len(neighbor1) != 1 or len(neighbor2) != 1:
            continue

        neighbor_idx1 = neighbor1[0].GetIdx()
        neighbor_idx2 = neighbor2[0].GetIdx()

        # Collect information for deleting dummy atoms and adding bonds
        reconnection_ops.append({
            'dummy_indices': (idx1, idx2),
            'neighbor_indices': (neighbor_idx1, neighbor_idx2),
            'bond_type': bond_type
        })

# Sort reconnection operations by dummy atom indices in descending order
reconnection_ops.sort(key=lambda x: max(x['dummy_indices']), reverse=True)

# Perform reconnection operations
for op in reconnection_ops:
    idx1, idx2 = op['dummy_indices']
    neighbor_idx1, neighbor_idx2 = op['neighbor_indices']
    bond_type = op['bond_type']

    # Remove dummy atoms in descending order
    for idx in sorted([idx1, idx2], reverse=True):
        mol_reconstructed.RemoveAtom(idx)

    # Add a bond to reconnect the molecule
    mol_reconstructed.AddBond(neighbor_idx1, neighbor_idx2, bond_type)

# Get the final molecule and sanitize
mol_final = mol_reconstructed.GetMol()
Chem.SanitizeMol(mol_final)

# Compare the original and modified SMILES strings
smiles_original = Chem.MolToSmiles(mol)
smiles_modified = Chem.MolToSmiles(mol_final)

def clean_smiles_regex(smiles):
    # Use regular expressions to match and remove virtual atom labels of the form [number*]
    cleaned_smiles = re.sub(r'\[\d+\*\]', '', smiles)
    return cleaned_smiles

cleaned_smiles = clean_smiles_regex(smiles_modified)

print("Original SMILES:", smiles_original)
print("Resulted SMILES:", cleaned_smiles)
