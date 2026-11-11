"""Backend modules implement the global part of the ``K u = F`` workflow.

A backend module is selected from ``main.py`` with ``--backend``. Each module
should expose the following functions:

    allocate_global_system(number_of_nodes)
    assemble_global_system(global_lhs, global_rhs, element_lhs, element_rhs, elements)
    impose_dirichlet_conditions(global_lhs, global_rhs, constrained_dofs, prescribed_values)
    solve_linear_system(global_lhs, global_rhs)
    visualize(coordinates, elements, solution, scalar_name="u", show_edges=True, screenshot=None)

``impose_dirichlet_conditions`` should return a new system and leave its inputs
unchanged, because ``main.py`` uses the original system for a PDE residual check.

The local arrays come from ``assembly.assemble_element_systems``:

    element_lhs: (n_elements, n_local, n_local)
    element_rhs: (n_elements, n_local)

The dense reference backend is a private baseline. The starter modules are
boiler plates for student implementations.
"""
