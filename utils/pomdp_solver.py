import subprocess
import shutil
def generate_policy(path,pomdp_file='program.pomdp', timeout=5, policy_file='program.policy'):

	#path='/home/saeid/software/sarsop/src/pomdpsol'
	print (policy_file)
	timeout = 20
	subprocess.check_output([path, pomdp_file, \
	            '--timeout', str(timeout), '--output', policy_file])

	shutil.move('program.policy','model/')