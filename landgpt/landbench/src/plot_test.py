import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
import os
from pathlib import Path
import numpy as np
from config import get_args
from path_utils import resolve_root, resolve_mask_path, lat_filename, lon_filename
# ---------------------------------# ---------------------------------

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new

def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):] 
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)] 
  return x_new

# configures
cfg = get_args()

def _resolve_base_path(cfg):
    return resolve_root(cfg['inputs_path'], cfg['product'], cfg['spatial_resolution'])

base_path = _resolve_base_path(cfg)
PATH = base_path.as_posix().rstrip('/') + '/'
mask = np.load(resolve_mask_path(base_path, cfg['spatial_resolution']))
mask = two_dim_lon_transform(mask)

work_root = base_path / cfg['workname']
focast_dir = 'focast_time ' + str(cfg['forcast_time'])

def _model_path(name: str) -> str:
    return (work_root / name / focast_dir).as_posix().rstrip('/') + '/'

out_path = _model_path(cfg['modelname'])
y_pred = np.load(out_path+'_predictions.npy')

y_pred = lon_transform(y_pred)

out_path_process = _model_path('Process')
fig_dir = Path(out_path) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

ALL_FIGURES = ['Fig.1', 'Fig.2', 'Fig.3', 'Fig.4', 'Fig.5', 'Fig.6', 'Fig.7', 'Fig.8']
PLOT_ALIASES = {
    'box': ['Fig.1'],
    'spatial': ['Fig.2', 'Fig.3', 'Fig.4', 'Fig.5', 'Fig.7', 'Fig.8'],
    'metrics': ['Fig.2', 'Fig.3', 'Fig.4', 'Fig.5', 'Fig.7', 'Fig.8'],
    'timeseries': ['Fig.6'],
}


def _resolve_requested_sections(raw):
    if not raw:
        return ALL_FIGURES
    normalized = []
    lowered = [token.lower() for token in raw]
    if any(token in ('all', '*') for token in lowered):
        return ALL_FIGURES
    for token in raw:
        token_clean = token.strip()
        if not token_clean:
            continue
        lower = token_clean.lower()
        if lower in PLOT_ALIASES:
            for sec in PLOT_ALIASES[lower]:
                if sec not in normalized:
                    normalized.append(sec)
            continue
        if lower.startswith('fig'):
            suffix = lower.split('fig', 1)[1].lstrip('. ')
            key = f"Fig.{suffix}"
            if key in ALL_FIGURES and key not in normalized:
                normalized.append(key)
            continue
    return normalized or ALL_FIGURES


plot_sections = _resolve_requested_sections(cfg.get('plot_figures'))
plot_sections_set = set(plot_sections)
print('[plot_test] plotting sections:', ', '.join(plot_sections))


def save_and_show(tag):
    fig = plt.gcf()
    manager = getattr(fig.canvas, "manager", None)
    if manager and hasattr(manager, "set_window_title"):
        try:
            manager.set_window_title(tag)
        except Exception:
            pass
    filename = f"{cfg['modelname']}_{tag}.png"
    file_path = fig_dir / filename
    fig.savefig(file_path.as_posix(), dpi=300, bbox_inches='tight')
    print(f"[plot_test] saved {tag}: {file_path}")
    plt.show()

name_pred = cfg['modelname']
if cfg['modelname'] in ["Process"] and cfg['label'] == ["volumetric_soil_water_20cm"]:
	y_pred = (y_pred[1:])/(1000)
	y_test = np.load(out_path+'observations.npy')
elif cfg['modelname'] in ["Process"] and cfg['label'] == ["surface_sensible_heat_flux"]:
	y_test = np.load(out_path+'observations.npy')
	y_pred = -(y_pred[1:])/(86400*cfg['forcast_time'])

else:
	y_test = np.load(out_path+'observations.npy')
y_test = lon_transform(y_test) 

	#y_pred = lon_transform(y_pred) 
print('y_pred is',y_pred[0])




mask[-int(mask.shape[0]/5.4):,:]=0
min_map = np.min(y_test,axis=0)
max_map = np.max(y_test,axis=0)
mask[min_map==max_map] = 0

name_test = 'Observations'
pltday =  135 # used for plt spatial distributions at 'pltday' day
#np.savetxt("/data/test/y_test.csv",y_test[0],delimiter=",")
r2_  = np.load(out_path+'r2_'+cfg['modelname'] +'.npy')
r2_ = two_dim_lon_transform(r2_) 
r_  = np.load(out_path+'r_'+cfg['modelname'] +'.npy')
r_ = two_dim_lon_transform(r_) 
urmse_  = np.load(out_path+'urmse_'+cfg['modelname'] +'.npy')
urmse_ = two_dim_lon_transform(urmse_) 
rmse_  = np.load(out_path+'rmse_'+cfg['modelname'] +'.npy')
rmse_ = two_dim_lon_transform(rmse_) 
bias_  = np.load(out_path+'bias_'+cfg['modelname'] +'.npy')
bias_ = two_dim_lon_transform(bias_) 




# reuse PATH if needed later

lat_file_name = lat_filename(cfg['spatial_resolution'])
lon_file_name = lon_filename(cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(base_path / lat_file_name)
lon_ = np.load(base_path / lon_file_name)
lon_ = np.linspace(-180,179,int(y_pred.shape[2]))
#print(lon_)
# Figure 6： configure for time series plot
#sites_lon_index=[120,80,220,280,270]
#sites_lat_index=[110,40,50,55,60]
sites_lon_index=[100,100,100,100,100]
sites_lat_index=[55,50,45,43,42]
if 'Fig.6' in plot_sections_set and cfg['label'] == ["surface_sensible_heat_flux"]:
	y_pred_process = np.load(out_path_process+'_predictions.npy')
	y_pred_process = lon_transform(y_pred_process)
	y_pred_process = -(y_pred_process[1:])/(86400*cfg['forcast_time'])
# ---------------------------------
# Staitic 1：R2,ubrmse
# ---------------------------------
mask_data = r2_[mask==1]
total_data = mask_data.shape[0]
#print('total_data  shape is', total_data.shape)
sea_nannum = np.sum(mask==0)
r_nannum = np.isnan(r_).sum()
print('the r NAN numble of',cfg['modelname'],'model is :',r_nannum-sea_nannum)
print('the average r2 of',cfg['modelname'],'model is :',np.nanmedian(r2_[mask==1]))
print('the average ubrmse of',cfg['modelname'],'model is :',np.nanmedian(urmse_[mask==1]))
print('the average r of',cfg['modelname'],'model is :',np.nanmedian(r_[mask==1]))
print('the average rmse of',cfg['modelname'],'model is :',np.nanmedian(rmse_[mask==1]))
print('the average bias of',cfg['modelname'],'model is :',np.nanmedian(bias_[mask==1]))
# ---------------------------------
# Figure 1： box plot
# ---------------------------------
if 'Fig.1' in plot_sections_set:
	# r2
	fig = plt.figure(figsize=(4.5, 4))
	r2_box = r2_[mask==1]
	r2_box = r2_box[np.isfinite(r2_box)]
	data_r2 = [r2_box]
	ax = plt.subplot(111)
	plt.ylabel('R$^{2}$')
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.spines['right'].set_linewidth(2)
	ax.spines['top'].set_linewidth(2)
	ax.boxplot(
            data_r2,
            notch=True,
            patch_artist=True,
            showfliers=False,
            tick_labels=[cfg['modelname']],
            boxprops=dict(facecolor='lightblue', color='black'),
        )
	save_and_show('fig1_r2_box')

	# urmse
	fig = plt.figure(figsize=(4.5, 4))
	urmse_box = urmse_[mask==1]
	urmse_box = urmse_box[np.isfinite(urmse_box)]
	data_urmse = [urmse_box]
	ax = plt.subplot(111)
	plt.ylabel("urmse")
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.spines['right'].set_linewidth(2)
	ax.spines['top'].set_linewidth(2)
	ax.boxplot(
            data_urmse,
            notch=True,
            patch_artist=True,
            showfliers=False,
            tick_labels=[cfg['modelname']],
            boxprops=dict(facecolor='red', color='black'),
        )
	save_and_show('fig1_urmse_box')

	# r
	fig = plt.figure(figsize=(4.5, 4))
	r_box = r_[mask==1]
	r_box = r_box[~np.isnan(r_box)]
	data_r = [r_box]
	ax = plt.subplot(111)
	plt.ylabel("r")
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.spines['right'].set_linewidth(2)
	ax.spines['top'].set_linewidth(2)
	ax.boxplot(
            data_r,
            notch=True,
            patch_artist=True,
            showfliers=False,
            tick_labels=[cfg['modelname']],
            boxprops=dict(facecolor='green', color='black'),
        )
	save_and_show('fig1_r_box')
	print('Figure 1 : box plot completed!')

# ------------------------------------------------------------------
# Figure 2： spatial distributions for predictions and observations
# ------------------------------------------------------------------
if 'Fig.2' in plot_sections_set:
	plt.figure(figsize=(12, 5))
	plt.subplot(1,2,1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,179.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	print('--------------------')
	print(xi)
	y_pred_pltday = y_pred[pltday, :,:]
	y_pred_pltday[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_20cm"]:
		cs = m.contourf(xi,yi, y_pred_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('m$^{3}$/m$^{3}$')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, y_pred_pltday, np.arange(-140, 141, 20), cmap='jet') 
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('W/m$^{2}$') 
	plt.title(name_pred)

	# observations
	plt.subplot(1,2,2)
	m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90.,18.)
	meridians = np.arange(-180.,179.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	print('xi is',xi)
	y_test_pltday = y_test[pltday, :,:]
	y_test_pltday[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_20cm"]:
		cs = m.contourf(xi,yi, y_test_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('m$^{3}$/m$^{3}$')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, y_test_pltday, np.arange(-140, 141, 20), cmap='jet')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('W/m$^{2}$')
	plt.title(name_test)
	print('Figure 2 : spatial distributions for predictions and observations completed!')
	save_and_show('fig2_pred_obs')
# ------------------------------------------------------------------
# Figure 3： spatial distributions for r2
# ------------------------------------------------------------------
if 'Fig.3' in plot_sections_set:
	plt.figure(figsize=(12, 5))
	plt.subplot(1,2,1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	r2_[mask==0]=-9999
	cs = m.contourf(xi,yi, r2_, np.arange(-1, 1, 0.1), cmap='seismic')  
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('R$^{2}$')
	plt.title(name_pred)

	plt.subplot(1,2,2)
	r2_mask_ = np.zeros(r2_.shape)
	r2_mask_[np.isnan(r2_)] = 1
	r2_mask_[mask==0]=0
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	cs = m.contourf(xi,yi, r2_mask_, np.arange(0, 1.5,0.5), cmap='bwr') #'seismic' 
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('R$^{2} NAN ("1" is NAN in land region )')
	plt.title(name_pred)

	print('Figure 3: spatial distributions for r2 completed!')
	save_and_show('fig3_r2_spatial')

# ------------------------------------------------------------------
# Figure 4： spatial distributions for ubrmse
# ------------------------------------------------------------------
if 'Fig.4' in plot_sections_set:
	plt.figure(figsize=(8, 4))
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)

	# convlstm
	urmse_[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_20cm"]:
		cs = m.contourf(xi,yi, urmse_, np.arange(0, 0.2, 0.01), cmap='RdBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, urmse_, np.arange(0, 51, 5), cmap='RdBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('ubrmse(W/m$^{2}$)')
	plt.title(name_pred)
	print('Figure 4: spatial distributions for ubrmse completed!')
	save_and_show('fig4_ubrmse_spatial')

# ------------------------------------------------------------------
# Figure 5： spatial distributions for r
# ------------------------------------------------------------------
if 'Fig.5' in plot_sections_set:
	plt.figure(figsize=(12, 5))
	plt.subplot(1,2,1)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	r_[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_20cm"]:
		cs = m.contourf(xi,yi, r_, np.arange(0, 1.1,0.1), cmap='jet') #'seismic' 
		cbar = m.colorbar(cs, location='bottom', pad="10%")
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, r_, np.arange(0, 1.1,0.1), cmap='jet') #'seismic' 
		cbar = m.colorbar(cs, location='bottom', pad="10%")

	cbar.set_label('R')
	plt.title(name_pred)

	plt.subplot(1,2,2)
	r_mask_ = np.zeros(r_.shape)
	r_mask_[np.isnan(r_)] = 1
	#r_mask_[mask==0]=0
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	cs = m.contourf(xi,yi, r_mask_, np.arange(0, 1.5,0.5), cmap='bwr') #'seismic' 
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('R NAN ("1" is NAN in land region )')
	plt.title(name_pred)

	print('Figure 5: spatial distributions for r completed!')
	save_and_show('fig5_r_spatial')

# ---------------------------------
# Figure 6： time series plot
# ---------------------------------
if 'Fig.6' in plot_sections_set:
	plt.figure(figsize=(8, 4))
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	for lon_index,lat_index in zip(sites_lon_index,sites_lat_index):
		lon_idx = int(np.clip(lon_index, 0, len(lon_)-1))
		lat_idx = int(np.clip(lat_index, 0, len(lat_)-1))
		lon=lon_[lon_idx]
		lat=lat_[lat_idx]
		plt.plot(lon, lat, marker='*', color='red', markersize=9)
	plt.legend(loc=0)
	save_and_show('fig6_location_map')

	data_all = [y_test,y_pred,y_pred]#y_pred_process
	color_list=['black','blue','blue']#red
	for lon_index,lat_index in zip(sites_lon_index,sites_lat_index):
		count=0
		fig, axs = plt.subplots(1,1,figsize=(15, 2))
		lon_idx = int(np.clip(lon_index, 0, len(lon_)-1))
		lat_idx = int(np.clip(lat_index, 0, len(lat_)-1))
		print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[lat_idx],ln_v=lon_[lon_idx]))
		print('r is',r_[lat_idx,lon_idx]) 
		print('urmse is', urmse_[lat_idx,lon_idx]) 
		print('rmse is',rmse_[lat_idx,lon_idx]) 
		print('bias is', bias_[lat_idx,lon_idx]) 
		for data_f5plt in (data_all):        
			axs.plot(data_f5plt[:,lat_idx,lon_idx], color=color_list[count])#label=name_plt5[count]
			axs.legend(loc=1)
			count = count+1

		axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[lat_idx],ln_v=lon_[lon_idx]))
		lat_val = float(lat_[lat_idx])
		lon_val = float(lon_[lon_idx])
		tag = f"fig6_timeseries_lat{lat_val:.2f}_lon{lon_val:.2f}".replace('.', 'p').replace('-', 'm')
		save_and_show(tag)
	print('Figure 6：time series plot completed!')
# ------------------------------------------------------------------
# Figure 7： spatial distributions for bias
# ------------------------------------------------------------------
if 'Fig.7' in plot_sections_set:
	plt.figure(figsize=(12, 5))
	plt.subplot(1,2,1)
	bias_ = np.mean((y_pred-y_test),axis=0)
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	# convlstm
	bias_[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_20cm"]:
		cs = m.contourf(xi,yi, bias_, np.arange(-0.04, 0.05,0.01), cmap='coolwarm') #'seismic' 
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('bias(m$^{3}$/m$^{3}$)')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, bias_, np.arange(-32, 33, 8), cmap='coolwarm') #'seismic' 
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('bias(W/m$^{2}$)')
	plt.title(name_pred)

	plt.subplot(1,2,2)
	r_mask_ = np.zeros(r_.shape)
	r_mask_[np.isnan(r_)] = 1
	r_mask_[mask==0]=0
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)
	cs = m.contourf(xi,yi, r_mask_, np.arange(0, 1.5,0.5), cmap='bwr') #'seismic' 
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	cbar.set_label('bias NAN ("1" is NAN in land region )')
	plt.title(name_pred)

	#plt.savefig(out_path + 'r_'+ cfg['modelname'] + '_spatial distributions.png')
	print('Figure 7: spatial distributions for bias completed!')
	save_and_show('fig7_bias_spatial')
# ------------------------------------------------------------------
# Figure 4： spatial distributions for rmse
# ------------------------------------------------------------------
if 'Fig.8' in plot_sections_set:
	plt.figure(figsize=(8, 4))
	lon, lat = np.meshgrid(lon_, lat_)
	m = Basemap()
	m.drawcoastlines()
	m.drawcountries()
	parallels = np.arange(-90.,90,18.)
	meridians = np.arange(-180.,180.,36.)
	m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
	m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
	xi, yi = m(lon, lat)

	# convlstm
	urmse_[mask==0]=-9999
	if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"]  or cfg['label'] == ["volumetric_soil_water_layer_20"]:
		cs = m.contourf(xi,yi, rmse_, np.arange(0, 0.2, 0.01), cmap='RdBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('rmse(m$^{3}$/m$^{3}$)')
	elif cfg['label'] == ["surface_sensible_heat_flux"]:
		cs = m.contourf(xi,yi, rmse_, np.arange(0, 51, 5), cmap='RdBu')  
		cbar = m.colorbar(cs, location='bottom', pad="10%")
		cbar.set_label('rmse(W/m$^{2}$)')
	plt.title(name_pred)
	print('Figure 8: spatial distributions for rmse completed!')
	save_and_show('fig8_rmse_spatial')
