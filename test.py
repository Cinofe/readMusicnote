import math
def reward_function(params):
    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    DIRECTION_THRESHOLD = 10.0
    ABS_STEERING_THRESHOLD = 30

    on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle'])
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints'] 
    heading = params['heading']

    reward = math.exp(-6 * distance_from_center)

    def on_track_reward(current_reward, on_track):
        if not on_track:
            current_reward = MIN_REWARD
        else:
            current_reward = MAX_REWARD
        return current_reward

    def distance_from_center_reward(current_reward, track_width, distance_from_center):
        marker_1 = 0.1 * track_width
        marker_2 = 0.25 * track_width
        marker_3 = 0.5 * track_width

        if distance_from_center <= marker_1:
            current_reward *= 1.2
        elif distance_from_center <= marker_2:
            current_reward *= 0.8
        elif distance_from_center <= marker_3:
            current_reward += 0.5
        else:
            current_reward = MIN_REWARD

        return current_reward

    def straight_line_reward(current_reward, steering, speed):
        if abs(steering) < 0.1 and speed > 3:
            current_reward *= 1.2
        return current_reward

    def direction_reward(current_reward, waypoints, closest_waypoints, heading):
        next_point = waypoints[closest_waypoints[1]]
        prev_point = waypoints[closest_waypoints[0]]

        direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) 
        direction = math.degrees(direction)
        
        if direction_diff > DIRECTION_THRESHOLD:
            current_reward *= 0.5

        return current_reward

    def steering_reward(current_reward, steering):
        if abs(steering) > ABS_STEERING_THRESHOLD:
            current_reward += 0.8
        return current_reward

    def throttle_reward(current_reward, speed, steering):
        if speed > 2.5 - (0.4 * abs(steering)):
            current_reward *= 0.8
        return current_reward

    reward = on_track_reward(reward, on_track)
    reward = distance_from_center_reward(reward, track_width, distance_from_center)
    reward = straight_line_reward(reward, steering, speed)
    reward = direction_reward(reward, waypoints, closest_waypoints, heading)
    reward = steering_reward(reward, steering)
    reward = throttle_reward(reward, speed, steering)

    return float(reward)