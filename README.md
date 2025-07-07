# Band Descriptor for Spatial Cellular Analysis in Tissue Samples

## Abstract

Human tissue samples exhibit remarkable cellular and structural diversity, where alterations in the spatial arrangement of cells can signal the onset or progression of disease. Therefore, characterizing these spatial cellular interactions and linking them to clinical endpoints is critical to advance our understanding of disease biology and improve patient care. 

In this work, we introduce a **band descriptor** that quantifies the local neighborhood of each cell by computing the relative abundance of neighboring cell types using concentric bands. 

We demonstrate the efficacy of our approach by highlighting two key benefits:
- It enables the unsupervised discovery of **spatiotypes** (substructures defined by local cellular configurations).
- It provides an explicit encoding of spatial context in cell-level graphs — capturing long-range cell interactions across tissue.

Our experiments in a lung tissue cohort reveal distinct spatial patterns of cellular arrangement that differentiate control from disease samples and may also reflect disease progression (unaffected, less affected, or more affected). Furthermore, by explicitly modeling spatial context, our band descriptor enhances node-level representations, enabling an end-to-end Graph Neural Network (GNN) to achieve high accuracy in a clinical prediction task with fewer layers. This reduction in network depth decreases over-smoothing and improves interpretability, underscoring our approach's potential for broad adoption in tissue-based studies and clinical applications.

---

## Concept Diagram

<!-- Include your concept diagram below -->
![Concept Diagram](./assets/concept_diagram.png)

---

## Documentation

To be added soon along with source code.

## Authors

- Muhammad Dawood<sup>1,2</sup> [[email]](mailto:muhammad.dawood@ndcls.ox.ac.uk)  
- Emily Thomas<sup>1,2</sup>  
- Rosalin Cooper<sup>1,3</sup>  
- Carlo Pescia<sup>4</sup>  
- Anna Sozanska<sup>1</sup>  
- Hosuk Ryou<sup>1,2</sup>  
- Daniel Royston<sup>1,3</sup>  
- Jens Rittscher<sup>2</sup>  

<sup>1</sup> Nuffield Division of Clinical Laboratory Sciences, University of Oxford, UK  
<sup>2</sup> Institute of Biomedical Engineering, University of Oxford, UK  
<sup>3</sup> Oxford University Hospitals, Oxford, UK  
<sup>4</sup> Division of Pathology, ASST Santi Paolo e Carlo, Milan, Italy  

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
