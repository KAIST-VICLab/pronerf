mkdir -p data
cd data
if [ ! -f nerf_example_data.zip ]; then
  wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
fi
unzip -n nerf_example_data.zip
cd ..
