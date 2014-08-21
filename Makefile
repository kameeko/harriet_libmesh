# $Id: Makefile,v 1.8 2008-10-16 18:04:01 roystgnr Exp $

# The location of the mesh library
LIBMESH_DIR = ~/software/libmesh_build

# include the library options determined by configure
include $(LIBMESH_DIR)/Make.common

target 	   := ./steady_convdiff-$(METHOD)

# include my common application options
include Make_template

