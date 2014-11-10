#!/bin/sh

GRINS_RUN=${GRINS_RUN:-$LIBMESH_RUN}
GRINS_SOLVER_OPTIONS=${GRINS_SOLVER_OPTIONS:-$LIBMESH_OPTIONS}

$GRINS_RUN /home/kameeko/software/grins_build_practice/bin/grins /home/kameeko/kameeko/course16/grad/dmd/harriet_libmesh/grins/practice_ins/practice_convdiff.in $GRINS_SOLVER_OPTIONS
