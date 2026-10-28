import numpy as np
import matplotlib.pyplot as plt
from analyb.histogram import Hist1D
from analyb.comparison import artifact_comparison_1d
from analyb.makelatex import write_compact_plotbook

def test_separate_ratio_uncertainties():
 a=Hist1D([6,2,3],[0,1,3,4],errors=[2,1,1])
 b=Hist1D([3,4,0],[0,1,3,4],errors=[1,2,0])
 fig,axes,d=artifact_comparison_1d(a,b,points='reference',separate_ratio_errors=True)
 np.testing.assert_allclose(d.ratio[:2],[2,.5]);assert np.isnan(d.ratio[2])
 np.testing.assert_allclose(axes[1].patches[0].get_data().values[:2],[2,.5])
 segments=axes[1].containers[0].lines[2][0].get_segments()
 np.testing.assert_allclose(segments[0][:,1],[2/3,4/3])
 np.testing.assert_allclose(segments[1][:,1],[.5,1.5])
 assert not axes[0].get_title();plt.close(fig)

def test_compact_book_pages(tmp_path):
 p=tmp_path/'book.tex'
 write_compact_plotbook(p,{'Particle':[(f'{i}.pdf',f'Jet {i}') for i in range(6)],'Reco':[('r.pdf','Reco')]})
 text=p.read_text();assert text.count('\\clearpage')==2
 assert text.count('\\includegraphics')==7

def test_cutflow_signed_weights_and_absent_reference_errors():
 from analyb.cutflow import cutflow_comparison
 fig,(main,ratio)=cutflow_comparison(['Input','Cut'],[4.,-2.],[2.,1.],numerator_errors=[3.,2.],weighted=True)
 np.testing.assert_allclose(ratio.patches[0].get_data().values,[2.,-2.])
 # The reference has no errorbar collection when uncertainty was not supplied.
 assert not ratio.containers[0].has_yerr
 assert ratio.get_ylim()[0] < -2.
 plt.close(fig)


def test_hist2d_nonuniform_geometry_and_orientation():
 from analyb.plot import hist2d_plot
 from matplotlib.colors import SymLogNorm
 counts=np.array([[1.,2.,3.],[4.,-1.,6.]])
 x=np.array([0.,1.,4.]);y=np.array([0.,2.,3.,9.])
 norm=SymLogNorm(linthresh=1.,vmin=-1.,vmax=6.)
 fig,ax=hist2d_plot((counts,x,y),norm=norm,colorbar_label='Weighted events')
 mesh=ax.collections[0];coords=mesh.get_coordinates()
 np.testing.assert_array_equal(coords[0,:,0],x)
 np.testing.assert_array_equal(coords[:,0,1],y)
 np.testing.assert_array_equal(mesh.get_array().reshape(3,2),counts.T)
 assert mesh.norm is norm
 assert fig.axes[1].get_ylabel()=='Weighted events'
 plt.close(fig)


def test_compact_three_rows_and_section_override(tmp_path):
 p=tmp_path/'dense.tex'
 write_compact_plotbook(p,{'Plots':[(f'{i}.pdf','x') for i in range(8)],'Native':[(f'n{i}.pdf','x') for i in range(6)]},rows_per_page=3,section_rows={'Native':2})
 text=p.read_text()
 assert text.count('\\clearpage')==3
 assert text.count('height=0.27')==8
 assert text.count('height=0.405')==6

def test_native_cutflow_retains_all_signed_values():
 from analyb.cutflow import native_cutflow_chart
 fig,ax=native_cutflow_chart(['initial','upstream','final'],[9.,-1.,2.])
 np.testing.assert_array_equal([p.get_width() for p in ax.patches],[9.,-1.,2.])
 assert len(ax.get_yticklabels())==3
 assert len(ax.containers)==1  # no invented uncertainty bars
 plt.close(fig)
