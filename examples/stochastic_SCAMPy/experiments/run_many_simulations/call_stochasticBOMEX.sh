# Allows use of tr in Mac
export LC_CTYPE=C
# Recover name of SCAMPy directory
scm_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Simulation names
declare -a simnames=("StochasticBomex")
# Values to be modified
val_prev_param=0.01
uuid_prev=51515

num_params=$#/2  # `$#` fetches no. cmd-line-args. Divide by 2 since we input (see `run_SCAMPY`) `u` and `u_names`
suffix=""
allparams=""
for (( i=1; i<=$num_params; i++ ))
do
    j=$((num_params+i))
    suffix=${suffix}_${!i}
    allparams=${allparams}${!i}
    echo ${!j}
    echo ${!i}
done

# Loop over simulations
for simname in "${simnames[@]}"
do
    # Create directory to store input files
    scampy_dir="sim_${simname}${suffix}"
    scampy_paramlist="${scampy_dir}/paramlist_${simname}.in"
    scampy_namelist="${scampy_dir}/${simname}.in"
    mkdir ${scampy_dir}
    cp Output.${simname}.00000/paramlist_${simname}.in ${scampy_paramlist}
    cp Output.${simname}.00000/${simname}.in ${scampy_namelist}
    cp Output.${simname}.00000/paramlist_${simname}.in paramlist_${simname}.in
    cp Output.${simname}.00000/${simname}.in ${simname}.in

    # Modify parameters from input files
    for (( i=1; i<=$num_params; i++ ))
    do
        j=$((num_params+i))  # fetch name parameter (j=index of input argument)
        line_param=$( awk '/"'${!j}'/{print NR}' ${scampy_paramlist} )
        gawk  'NR=='$line_param'{gsub(/'$val_prev_param'/,'"${!i}"')};1' ${scampy_paramlist} > tmp${suffix} && mv tmp${suffix} ${scampy_paramlist}
    done

    # Generate random 5-digit number to use as UUID
    uuid=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1)
    line_uuid=$( awk '/"uuid/{print NR}' ${scampy_namelist} )
    awk 'NR==row_num {sub(val,val2)};1' row_num="$line_uuid" val="$uuid_prev" "val2=$uuid" ${scampy_namelist} > tmp_${uuid} && mv tmp_${uuid} ${scampy_namelist}

    output_dir=$(awk -F"output_root" '/output_root/{print $2}' ${scampy_namelist})
    output_dir=$(echo "$output_dir" | sed 's|[": ]||g')
    full_output_dir="${output_dir}Output.${simname}.${uuid}"
    output_paramlist="${full_output_dir}/paramlist_${simname}.in"
    output_namelist="${full_output_dir}/${simname}.in"

    # Run SCAMPy with modified parameters
    conda run -n scampy python ${scm_dir}/main.py ${scampy_namelist} ${scampy_paramlist}
    echo "simulation done. uuid: ${uuid}. output dir: '${full_output_dir}'"
    
    # Copy used input files to output directory, since the copied files by SCAMPy are not the ones that are used.
    cp ${scampy_paramlist} ${output_paramlist}
    cp ${scampy_namelist} ${output_namelist}
    # echo "$(ls ${scm_dir})"
    rm -r ${scampy_dir}
    rm paramlist_${simname}.in ${simname}.in
    echo ${full_output_dir} >> ${allparams}.txt
done

