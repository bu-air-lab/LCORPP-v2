# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:35:21 2020

@author: cckklt
"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append('../carla')
except IndexError:
    pass


import carla

import argparse
import logging
import random
import copy
from agents.navigation import basic_agent
import time


def main():
    listen_count=0
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    
    argparser.add_argument(
        '-c2','--car2',
        action='store_true'
                )
    
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client('127.0.0.1', 2000)

    client.set_timeout(20.0)
    actor_list = []

    try:

        world = client.get_world()
        

        # --------------
        # Spawn ego vehicle
        # --------------
        blueprint=world.get_blueprint_library()
        v1_bp = blueprint.find('vehicle.tesla.model3')
        v1_bp.set_attribute('role_name','ego1')
        
        v2_bp=blueprint.find('vehicle.audi.tt')
        v2_bp.set_attribute('role_name','ego2')
        
        v3_bp=blueprint.find('vehicle.audi.tt')
        v3_bp.set_attribute('role_name','ego3')
        print('Vehicles role_name is set')
             
        v1_bp.set_attribute('color','0,0,255')
        v2_bp.set_attribute('color','0,255,0')
        v3_bp.set_attribute('color','255,0,0')
        print('Vehicles color is set')

        spawn_points = world.get_map().get_spawn_points()           
        start_transform = spawn_points[0]
        x=start_transform.location.x
        y=start_transform.location.y-10
        z=start_transform.location.z
        start_rotation=carla.Rotation(start_transform.rotation.pitch,start_transform.rotation.yaw,start_transform.rotation.roll)
        
        x_left=x+5
        #Set transform of v1,v2,v3,v4
        x1=x+5
        y1=y-30
        z1=z
        vehicle1_location=carla.Location(x1,y1,z1)
        vehicle1_transform=carla.Transform(vehicle1_location,start_rotation)
                     
        x2=x1
        y2=y-20
        z2=z
        vehicle2_location=carla.Location(x2,y2,z2)
        vehicle2_transform=carla.Transform(vehicle2_location,start_rotation)
              
        x3=x+1.5
        y3=y-15
        z3=z
        vehicle3_location=carla.Location(x3,y3,z3)
        vehicle3_transform=carla.Transform(vehicle3_location,start_rotation)
        
        x4=x1
        y4=y-10
        z4=z
        vehicle4_location=carla.Location(x4,y4,z4)
        vehicle4_transform=carla.Transform(vehicle4_location,start_rotation)
        
        y_left2=y
        vehicle_left2_location=carla.Location(x_left,y_left2,z)
        vehicle_left2_transform=carla.Transform(vehicle_left2_location,start_rotation)
        
        
        x_agent=x+1.5
        y_agent=y-25
        z_agent=z
        vehicle_agent_location=carla.Location(x_agent,y_agent,z_agent)
        vehicle_agent_transform=carla.Transform(vehicle_agent_location,start_rotation)
        
        
        transform_list=[]
        
        variation=random.randint(0,3)-random.randint(0,3)
        y_left=y-10-variation
        
        
        count=0
        for i in range(6):
              vehicle_location=carla.Location(x_left,y_left-i*5,z)
              vehicle_transform=carla.Transform(vehicle_location,start_rotation)
              transform_list.append(vehicle_transform)
        
        label=None
        num_vehicles=random.randint(1,2)
        sampled_list=random.sample(transform_list,num_vehicles)
        
        if num_vehicles>=3:
              label="1"
        
        elif num_vehicles==2:
              if abs(sampled_list[0].location.y-sampled_list[1].location.y)>=10:
                    label="0"
              else:
                    label='1'
        
        elif num_vehicles==1:
              label="0"
        
        print("Number of vehicles is",num_vehicles)
        print("Label is",label)
        
        for transform in sampled_list:
              vehicle=world.spawn_actor(v2_bp,transform)
              actor_list.append(vehicle)
              
        
        vehicle_agent=world.spawn_actor(v3_bp,vehicle_agent_transform)
        actor_list.append(vehicle_agent)
        print("vehicle_agent is spawned")
      
        # --------------
        # Place spectator on ego spawning
        # --------------

        spectator = world.get_spectator()
        spct_x,spct_y=x1,y1
        spct_z=4
        spct_location=carla.Location(spct_x,spct_y,spct_z)
        spct_transform=carla.Transform(spct_location,start_rotation)
        world_snapshot = world.wait_for_tick() 
        spectator.set_transform(spct_transform)
                
        #add the lidar sensor to agent vehicle
        lidar_bp=world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(40))
        lidar_bp.set_attribute('points_per_second',str(100000))
        lidar_bp.set_attribute('rotation_frequency',str(120))
        lidar_bp.set_attribute('range',str(20))
        lidar_bp.set_attribute('upper_fov',str(0))
        lidar_bp.set_attribute('lower_fov',str(-10))
        lidar_bp.set_attribute('sensor_tick',str(2.0))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar_sensor=world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle_agent)
        actor_list.append(lidar_sensor)
        
        time.sleep(1)
        def lidar_callback(point_cloud):
              if label=='0':
                    print(count,"Start to save")
                    point_cloud.save_to_disk('lidar_output_test/class_A/%.6d.ply' % point_cloud.frame)
                    #world.wait_for_tick()
              elif label=='1':
                    print(count,"Start to save")
                    point_cloud.save_to_disk('lidar_output_test/class_B/%.6d.ply' % point_cloud.frame)
              
                    #world.wait_for_tick()
        #lidar_sensor.listen(lambda point_cloud: point_cloud.save_to_disk('lidar_output/class_A/%.6d.ply' % point_cloud.frame) if label=='0' else point_cloud.save_to_disk('lidar_output/class_B/%.6d.ply' % point_cloud.frame))
        lidar_sensor.listen(lambda point_cloud:lidar_callback(point_cloud))
        
        #add the camera sensor
        """
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(1920))
        cam_bp.set_attribute("image_size_y",str(1080))
        cam_bp.set_attribute("fov",str(90))
        cam_location = carla.Location(0,0,2)
        cam_rotation = carla.Rotation(0,-45,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle_agent)
        actor_list.append(ego_cam)
        ego_cam.listen(lambda image: image.save_to_disk('Camera/output/%.6d.jpg' % image.frame))
        """




        #vehicle1.set_autopilot(False)
        #vehicle2.set_autopilot(False)
        #vehicle3.set_autopilot(False)
        vehicle_agent.set_autopilot(False)
        
        
        #vehicle.apply_control()
        while True:
              world_snapshot = world.wait_for_tick()
              time.sleep(1.5)
              lidar_sensor.stop()
              break


            
            
    finally:
        # --------------
        #client.stop_recorder()              
        vehicle_agent.destroy()
        for actor in actor_list:
              actor.destroy()
              print("destroyed")


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')