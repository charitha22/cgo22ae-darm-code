#!/usr/bin/python3
import sys
import re
from optparse import OptionParser


def handle_error(err_msg):
  print(f"ERROR : {err_msg}")
  sys.exit(-1)


def get_opt_level(compile_cmd):
  opt_level = re.findall(r" -O. ", compile_cmd)

  if(len(opt_level) != 1):
    handle_error("invalid opt level found in compile command!")

  return str(opt_level[0])


def get_target_triple(compile_cmd):

  target_triple = re.findall(r" -triple .* ", compile_cmd)

  if(len(target_triple) != 1):
    handle_error("invalid target triple found in compile command!")

  return target_triple[0].split(" ")[2]


def get_output_ir_fname(compile_cmd):

  output_fname = re.findall(r' -o .*\.ll ', compile_cmd)


  if(len(output_fname) != 1):
    handle_error("invalid output object file name found!")

  return str(output_fname[0]).split("-o ")[1]


def get_target_cpu(compile_cmd) :
  
  target_cpu = re.findall(r" -target-cpu .* ", compile_cmd)

  if(len(target_cpu) != 1):
    handle_error("invalid target triple found in compile command!")

  return target_cpu[0].split(" ")[2]


def gen_device_command_amd(cmd, options):
  opt_level = get_opt_level(cmd)
  
  emit_llvm_added = cmd.replace("-emit-obj", "-S -emit-llvm")
  
  if(options.run_O0_device):
    o0_optnone_added = emit_llvm_added.replace(opt_level, " -O0 " + "-disable-O0-optnone ")
  else:
    o0_optnone_added = emit_llvm_added.replace(opt_level, opt_level + "-disable-O0-optnone ")

  output_ext_changed = o0_optnone_added.replace(".o", ".ll")

  output_dir_changed = output_ext_changed.replace("/tmp/", options.out_loc + "/")
  
  return (output_dir_changed, opt_level)

def gen_device_command_cuda(cmd, options):
  opt_level = get_opt_level(cmd)
  
  emit_llvm_added = cmd.replace(" -S ", " -S -emit-llvm ")

  o0_optnone_added = emit_llvm_added.replace(opt_level, opt_level + "-disable-O0-optnone ")

  output_ext_changed = o0_optnone_added.replace(".s", ".ll")

  output_dir_changed = output_ext_changed.replace("/tmp/", options.out_loc + "/")
  
  return output_dir_changed

def gen_ptxas_command(command, ptxas_opt_level):
  curr_opt_level = get_opt_level(command)

  return command.replace(curr_opt_level, f" -{ptxas_opt_level} ")

def gen_pass_command(ir_fname, options):

  if options.disable_pass:
    opt_command = f"{options.llvm_home}/bin/opt {options.disable_pass_options}   -S < {ir_fname} > {options.out_loc}/after_pass.ll "
  else:
    if options.run_O0_device:
      opt_command = f"{options.llvm_home}/bin/opt {options.passes_before_cfmelder} -cfmelder {options.cfmelder_options} {options.passes_after_cfmelder} -S < {ir_fname} > {options.out_loc}/after_pass.ll "

    else:
      opt_command = f"{options.llvm_home}/bin/opt -cfmelder {options.cfmelder_options} -S < {ir_fname} > {options.out_loc}/after_pass.ll "

  return opt_command

def main() : 
  parser = OptionParser()
  parser.add_option("--llvm-home", help="LLVM home", dest="llvm_home", action="store")
  parser.add_option("--output-loc", help="location to write output files to", dest="out_loc", action="store")
  parser.add_option("--ptx-opt-level", help="optimization level for ptxas", dest="ptxas_opt_level", action="store")
  parser.add_option("--disable-pass", help="do not run the pass", dest="disable_pass", action="store_true", default=False)
  parser.add_option("--disable-pass-options", help="passes to run in baseline device ir", dest="disable_pass_options", action="store", default="")
  parser.add_option("--cfmelder-options", help="options to pass to cfmelder", dest="cfmelder_options", action="store", default="")
  parser.add_option("--run-O0-on-device", help="use O0 for device code", dest="run_O0_device", action="store_true", default=False)
  parser.add_option("--passes-before-cfmelder", help="additional options to run before cfmelder if O0 used for device", dest="passes_before_cfmelder", action="store", default="")
  parser.add_option("--passes-after-cfmelder", help="additional options to run after cfmelder if O0 used for device", dest="passes_after_cfmelder", action="store", default="")
  parser.add_option("--llc-options", help="additional options to llc", dest="llc_options", action="store", default="")
  (options, args) = parser.parse_args()


  compile_commands = []

  for line in sys.stdin:
    if line.startswith(" \""):
      # print(line.replace("\"",""))
      compile_commands.append(line.replace("\"", ""))

  instrumented_commands = []

  for command in compile_commands:
    if command.find(" -triple amdgcn-amd-amdhsa -aux-triple") != -1:

      (device_ir_gen_cmd, orig_opt_level) = gen_device_command_amd(command, options)

      ir_fname = get_output_ir_fname(device_ir_gen_cmd)
      obj_name = ir_fname.replace(".ll", ".o")
      opt_level = get_opt_level(device_ir_gen_cmd)
      target_triple = get_target_triple(device_ir_gen_cmd)
      target_cpu = get_target_cpu(device_ir_gen_cmd)

      
      instrumented_commands.append(device_ir_gen_cmd)
      
      opt_command = gen_pass_command(ir_fname, options)
      instrumented_commands.append(opt_command)

      if options.run_O0_device:
        llc_command = f"{options.llvm_home}/bin/llc {orig_opt_level} -mtriple {target_triple} -mcpu={target_cpu} -filetype=obj {options.llc_options} {options.out_loc}/after_pass.ll -o {obj_name}"

      else:
        llc_command = f"{options.llvm_home}/bin/llc {opt_level} -mtriple {target_triple} -mcpu={target_cpu} -filetype=obj {options.llc_options} {options.out_loc}/after_pass.ll -o {obj_name}"
      instrumented_commands.append(llc_command)

    elif command.find("-cc1 -triple nvptx64-nvidia-cuda") != -1:
      device_ir_gen_cmd = gen_device_command_cuda(command, options)

      ir_fname = get_output_ir_fname(device_ir_gen_cmd)
      obj_name = ir_fname.replace(".ll", ".s")
      opt_level = get_opt_level(device_ir_gen_cmd)
      target_triple = get_target_triple(device_ir_gen_cmd)
      target_cpu = get_target_cpu(device_ir_gen_cmd)

      
      instrumented_commands.append(device_ir_gen_cmd)

      opt_command = gen_pass_command(ir_fname, options)        
      instrumented_commands.append(opt_command)

      llc_command = f"{options.llvm_home}/bin/llc {opt_level} -mtriple {target_triple} -mcpu={target_cpu} {options.out_loc}/after_pass.ll -o {obj_name}"
      instrumented_commands.append(llc_command)

    elif command.find("ptxas") != -1 and options.ptxas_opt_level != None:
      ptx_command = gen_ptxas_command(command, options.ptxas_opt_level)
      instrumented_commands.append(ptx_command.replace("/tmp/", f"{options.out_loc}/"))
    else :
      instrumented_commands.append(command.replace("/tmp/", f"{options.out_loc}/"))

  for cmd in instrumented_commands:
    print(cmd)


      
if __name__ == "__main__" :
  main()