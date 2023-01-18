#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
from torch_models import SmallNet, MedNet
from time import sleep
import pandas as pd
import torch
import vizdoom as vzd
import REINFORCE

if __name__ == "__main__":

    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Makes the window appear (turned on by default)
    game.set_window_visible(False)

    # map and scenario
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(True)
    game.set_render_minimal_hud(True)  # If hud is enabled
    #game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed to use.
    # This can be done by adding buttons one by one:
    # game.clear_available_buttons()
    # game.add_available_button(vzd.Button.MOVE_LEFT)
    # game.add_available_button(vzd.Button.MOVE_RIGHT)
    # game.add_available_button(vzd.Button.ATTACK)
    # Or by setting them all at once:
    #game.set_available_buttons([vzd.Button.MOVE_LEFT,
    #                            vzd.Button.MOVE_RIGHT,
    #                            vzd.Button.MOVE_FORWARD,
    #                            vzd.Button.MOVE_BACKWARD,
    #                            vzd.Button.TURN_LEFT,
    #                            vzd.Button.TURN_RIGHT,
    #                            vzd.Button.ATTACK])

    game.set_available_buttons([vzd.Button.MOVE_LEFT,
                                vzd.Button.MOVE_RIGHT,
                                vzd.Button.MOVE_FORWARD,
                                vzd.Button.MOVE_BACKWARD,
                                vzd.Button.TURN_LEFT,
                                vzd.Button.TURN_RIGHT,
                                vzd.Button.ATTACK,
                                vzd.Button.CROUCH,
                                vzd.Button.JUMP,
                                vzd.Button.RELOAD,
                                vzd.Button.LOOK_DOWN,
                                vzd.Button.LOOK_UP,
                                #vzd.Button.SELECT_NEXT_WEAPON,
                                #vzd.Button.SELECT_PREV_WEAPON,
                                ])

    # Buttons that will be used can be also checked by:
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    # Adds game variables that will be included in state.
    # Similarly to buttons, they can be added one by one:
    # game.clear_available_game_variables()
    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
    # Or:
    game.set_available_game_variables([vzd.GameVariable.AMMO2])

    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)


    # Turns on the sound. (turned off by default)
    # game.set_sound_enabled(True)
    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
    # the sound is only useful for humans watching the game.

    # Sets the living reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console, in case of a problem this might provide additional information.
    #game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    n_actions = game.get_available_buttons_size()
    actions = []
    for n in range(n_actions):
        action_bool = [False]*n_actions
        action_bool[n] = True
        actions.append(action_bool)

    agent = REINFORCE.agent(n_actions=n_actions,
                            gradient_accumulation=16,
                            lr=2e-4,
                            model=SmallNet,
                            reward_scaling=100,
                            screen_dims=(game.get_screen_width(), game.get_screen_height())
                            )

    #agent.load_model('model30000.pt')

    # Run this many episodes
    episodes = 40000

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    #sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    sleep_time = 0.0
    reward_log = []
    for i in range(episodes):

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Which consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            screen_buf = torch.from_numpy(screen_buf).float()
            screen_buf = torch.nn.functional.pad(screen_buf, (0, 0, 0, 0, 4, 4), "constant", 0)  # pad so it fits the network
            screen_buf = torch.transpose(screen_buf, dim0=0, dim1=2)  # need to put channels first for VGG
            screen_buf = screen_buf.unsqueeze(0)/256  # add in batch dimension, required by VGG. Normalise

            action, log_probs = agent.act(screen_buf)
            r = game.make_action(actions[action])

            agent.remember((screen_buf, action, log_probs, r, game.is_episode_finished()))

            if sleep_time > 0:
                sleep(sleep_time)

        # make agent learn
        agent.update_networks()
        agent.reset()

        # Check how the episode went.
        print("Episode: {} | reward: {}".format(i, game.get_total_reward()))
        reward_log.append(game.get_total_reward())
        if i % 100 == 0:
            pd.Series(reward_log).to_csv("reward_log.csv")

        if i % 50000 == 0:
            agent.save_model("model_basic{}.pt".format(i))

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    agent.save_model("model_basic{}.pt".format(i)) # save final model
    game.close()