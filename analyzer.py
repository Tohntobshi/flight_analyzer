from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2 as cv
import sys
import numpy as np

if len(sys.argv) != 4:
    print('usage: analyzer.py <session_data> <session_video> <offset_ms>')
    exit(1)

_, data_file, video_file, offset = sys.argv

parsed_frames = []

cap = cv.VideoCapture(video_file)
while True:
    ret, frame = cap.read()
    if ret:
        parsed_frames.append(frame)
    else:
        break

cap.release()

parsed_timestamps = []
parsed_pitch_errors = []
parsed_roll_errors = []
parsed_pitch_error_ders = []
parsed_roll_error_ders = []
parsed_height_errors = []
parsed_height_error_ders = []
parsed_yaw_errors = []
parsed_yaw_error_ders = []
parsed_fl_vals = []
parsed_fr_vals = []
parsed_bl_vals = []
parsed_br_vals = []
parsed_freq_vals = []
parsed_pitch_err_ints = []
parsed_roll_err_ints = []
parsed_yaw_err_ints = []
parsed_height_err_ints = []
parsed_voltages = []
parsed_position_x_errors = []
parsed_position_y_errors = []
parsed_position_x_err_ders = []
parsed_position_y_err_ders = []
parsed_position_x_err_ints = []
parsed_position_y_err_ints = []

with open(data_file) as f:
    for line in f:
        time_ms, pitch_err, roll_err, pitch_err_der, roll_err_der, \
        height_err, height_err_der, yaw_err, yaw_err_der, \
        fl_mot, fr_mot, bl_mot, br_mot, freq, pitch_err_int, roll_err_int, yaw_err_int, height_err_int, \
        voltage, position_x_err, position_y_err, position_x_err_der, position_y_err_der, position_x_err_int, position_y_err_int = line.split(',')
        parsed_timestamps.append(int(time_ms))
        parsed_pitch_errors.append(float(pitch_err))
        parsed_roll_errors.append(float(roll_err))
        parsed_pitch_error_ders.append(float(pitch_err_der))
        parsed_roll_error_ders.append(float(roll_err_der))
        parsed_height_errors.append(float(height_err))
        parsed_height_error_ders.append(float(height_err_der))
        parsed_yaw_errors.append(float(yaw_err))
        parsed_yaw_error_ders.append(float(yaw_err_der))
        parsed_fl_vals.append(float(fl_mot))
        parsed_fr_vals.append(float(fr_mot))
        parsed_bl_vals.append(float(bl_mot))
        parsed_br_vals.append(float(br_mot))
        parsed_freq_vals.append(float(freq))
        parsed_pitch_err_ints.append(float(pitch_err_int))
        parsed_roll_err_ints.append(float(roll_err_int))
        parsed_yaw_err_ints.append(float(yaw_err_int))
        parsed_height_err_ints.append(float(height_err_int))
        parsed_voltages.append(float(voltage))
        parsed_position_x_errors.append(float(position_x_err))
        parsed_position_y_errors.append(float(position_y_err))
        parsed_position_x_err_ders.append(float(position_x_err_der))
        parsed_position_y_err_ders.append(float(position_y_err_der))
        parsed_position_x_err_ints.append(float(position_x_err_int))
        parsed_position_y_err_ints.append(float(position_y_err_int))

initial = parsed_timestamps[0]
parsedTimeStamps = list(map(lambda x: x - initial, parsed_timestamps))

padding = np.zeros([50], dtype='float32')
frames = np.array(parsed_frames)
timestamps = np.array(parsedTimeStamps, dtype='int32')
pitch_errors = np.concatenate((padding, np.array(parsed_pitch_errors, dtype='float32'), padding))
roll_errors = np.concatenate((padding, np.array(parsed_roll_errors, dtype='float32'), padding))
pitch_error_ders = np.concatenate((padding, np.array(parsed_pitch_error_ders, dtype='float32'), padding))
roll_error_ders = np.concatenate((padding, np.array(parsed_roll_error_ders, dtype='float32'), padding))
height_errors = np.concatenate((padding, np.array(parsed_height_errors, dtype='float32'), padding))
height_error_ders = np.concatenate((padding, np.array(parsed_height_error_ders, dtype='float32'), padding))
yaw_errors = np.concatenate((padding, np.array(parsed_yaw_errors, dtype='float32'), padding))
yaw_error_ders = np.concatenate((padding, np.array(parsed_yaw_error_ders, dtype='float32'), padding))
fl_vals = np.concatenate((padding, np.array(parsed_fl_vals, dtype='float32'), padding))
fr_vals = np.concatenate((padding, np.array(parsed_fr_vals, dtype='float32'), padding))
bl_vals = np.concatenate((padding, np.array(parsed_bl_vals, dtype='float32'), padding))
br_vals = np.concatenate((padding, np.array(parsed_br_vals, dtype='float32'), padding))
freq_vals = np.concatenate((padding, np.array(parsed_freq_vals, dtype='float32'), padding))
pitch_err_ints = np.concatenate((padding, np.array(parsed_pitch_err_ints, dtype='float32'), padding))
roll_err_ints = np.concatenate((padding, np.array(parsed_roll_err_ints, dtype='float32'), padding))
yaw_err_ints = np.concatenate((padding, np.array(parsed_yaw_err_ints, dtype='float32'), padding))
height_err_ints = np.concatenate((padding, np.array(parsed_height_err_ints, dtype='float32'), padding))
voltages = np.concatenate((padding, np.array(parsed_voltages, dtype='float32'), padding))
position_x_errors = np.concatenate((padding, np.array(parsed_position_x_errors, dtype='float32'), padding))
position_y_errors = np.concatenate((padding, np.array(parsed_position_y_errors, dtype='float32'), padding))
position_x_err_ders = np.concatenate((padding, np.array(parsed_position_x_err_ders, dtype='float32'), padding))
position_y_err_ders = np.concatenate((padding, np.array(parsed_position_y_err_ders, dtype='float32'), padding))
position_x_err_ints = np.concatenate((padding, np.array(parsed_position_x_err_ints, dtype='float32'), padding))
position_y_err_ints = np.concatenate((padding, np.array(parsed_position_y_err_ints, dtype='float32'), padding))

fig, ((ax_1, ax_2), (ax_3, ax_4), (ax_5, ax_6), (ax_7, ax_8), (ax_9, ax_10)) = plt.subplots(nrows=5, ncols=2)

frame_counter = 0
pause = False
plot_offset = int(offset)


def onkey_handler(event):
    global pause, plot_offset, frame_counter
    if event.key == ' ':
        pause = not pause
    if event.key == 'b':
        plot_offset += 1000
        print(f"plot offset {plot_offset}")
    if event.key == 'v':
        plot_offset += 10
        print(f"plot offset {plot_offset}")
    if event.key == 'c':
        plot_offset -= 10
        print(f"plot offset {plot_offset}")
    if event.key == 'x':
        plot_offset -= 1000
        print(f"plot offset {plot_offset}")
    if event.key == 'left':
        frame_counter -= 60
    if event.key == 'right':
        frame_counter += 60
    if event.key == 'up':
        frame_counter -= 1
    if event.key == 'down':
        frame_counter += 1


def get_ms_by_frame(n):  # video is supposed to be 60fps
    return int(n / 6) * 100 + (n % 6) * 16


def get_plot_index(ms):
    approx_index = int(ms / 11)
    approx_index = min(len(timestamps) - 1, max(0, approx_index))
    while timestamps[approx_index] < ms - 8 and approx_index + 1 < len(timestamps):
        approx_index += 1
    while timestamps[approx_index] - 8 > ms and approx_index - 1 >= 0:
        approx_index -= 1
    return approx_index


fig.canvas.mpl_connect("key_press_event", onkey_handler)
fig.canvas.set_window_title('Plots')
ax_1.axis('off')
ax_1.text(0, 0.5, "x, c - shift plot left;\nv, b - shift plot right;\nSPACE - play/stop;\narrows up, left - go backwards;\narrows down, right - go forward")
ax_2.axis('off')


def prepare_plot(ax, val_range, label):
    ax.set_ylim(-val_range, val_range)
    plot_line, = ax.plot(np.zeros([100], dtype='float32'), color='blue', alpha=0.8)
    ax.axvline(50, -val_range, val_range, linewidth=0.5, linestyle='--')
    ax.axhline(0, 0, 1, linewidth=0.5, linestyle='--')
    ax.set_title(label)
    ax.set_xticks([])
    text = ax.text(50, 0, "")
    return [plot_line, text]


def update_plot(artists, vals, text_val):
    plot_line, text = artists
    plot_line.set_ydata(vals)
    text.set_text(text_val)


def prepare_2_vals_plot(ax, val_range, label, label1, label2):
    ax.set_ylim(-val_range, val_range)
    plot_line_1, = ax.plot(np.zeros([100], dtype='float32'), color='blue', label=label1, alpha=0.8)
    plot_line_2, = ax.plot(np.zeros([100], dtype='float32'), color='red', label=label2, alpha=0.5)
    ax.legend()
    ax.axvline(50, -val_range, val_range, linewidth=0.5, linestyle='--')
    ax.axhline(0, 0, 1, linewidth=0.5, linestyle='--')
    ax.set_title(label)
    ax.set_xticks([])
    text = ax.text(50, 0, "")
    return [plot_line_1, plot_line_2, text]


def update_2_vals_plot(artists, vals_1, vals_2, text_val):
    plot_line_1, plot_line_2, text = artists
    plot_line_1.set_ydata(vals_1)
    plot_line_2.set_ydata(vals_2)
    text.set_text(text_val)


def prepare_point(ax, val_range, label):
    ax.set_title(label)
    ax.axis('equal')
    ax.axvline(0, -val_range, val_range, linewidth=0.5, linestyle='--')
    ax.axhline(0, 0, 1, linewidth=0.5, linestyle='--')
    point, = ax.plot([0], [0], marker='o')
    ax.set_xlim(-val_range, val_range)
    ax.set_ylim(-val_range, val_range)
    text = ax.text(-val_range * 0.5, 0, "")
    return [point, text]


def update_point(artists, x, y, text_val):
    point, text = artists
    point.set_xdata([x])
    point.set_ydata([y])
    text.set_text(text_val)


pitch_art = prepare_2_vals_plot(ax_3, 30, "pitch error", "val", "der")
roll_art = prepare_2_vals_plot(ax_4, 30, "roll error", "val", "der")
height_art = prepare_2_vals_plot(ax_5, 1, "height error", "val", "der")
yaw_art = prepare_2_vals_plot(ax_6, 180, "yaw error", "val", "der")
position_x_art = prepare_2_vals_plot(ax_7, 2, "position x error", "val", "der")
position_y_art = prepare_2_vals_plot(ax_8, 2, "position y error", "val", "der")
pitch_roll_err_art = prepare_point(ax_9, 30, "pitch roll error")
position_err_art = prepare_point(ax_10, 5, "position error")

def loop(i):
    global frame_counter, plot_offset, pause, frames, timestamps
    video_ms = get_ms_by_frame(frame_counter)
    plot_index = get_plot_index(video_ms - plot_offset)
    if frame_counter >= 0 and frame_counter < len(frames):
        cv.imshow("Video", frames[frame_counter])
    else:
        pause = True

    start_index = plot_index
    current_index = plot_index + 50
    end_index = plot_index + 100

    text = ax_2.text(0, 0.5, f"plot offset {plot_offset}ms\ntime {video_ms}ms\nfl {fl_vals[current_index]} fr {fr_vals[current_index]}\nbl {bl_vals[current_index]} br {br_vals[current_index]}")

    update_2_vals_plot(pitch_art, pitch_errors[start_index:end_index], pitch_error_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(pitch_errors[current_index], pitch_error_ders[current_index], pitch_err_ints[current_index]))
    update_2_vals_plot(roll_art, roll_errors[start_index:end_index], roll_error_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(roll_errors[current_index], roll_error_ders[current_index], roll_err_ints[current_index]))
    update_2_vals_plot(height_art, height_errors[start_index:end_index], height_error_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(height_errors[current_index], height_error_ders[current_index], height_err_ints[current_index]))
    update_2_vals_plot(yaw_art, yaw_errors[start_index:end_index], yaw_error_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(yaw_errors[current_index], yaw_error_ders[current_index], yaw_err_ints[current_index]))
    update_point(pitch_roll_err_art,  roll_errors[current_index], pitch_errors[current_index], "p {:.2f}\nr {:.2f}".format(pitch_errors[current_index], roll_errors[current_index]))
    update_point(position_err_art, position_x_errors[current_index], position_y_errors[current_index], "x {:.2f}\ny {:.2f}".format(position_x_errors[current_index], position_y_errors[current_index]))
    update_2_vals_plot(position_x_art, position_x_errors[start_index:end_index], position_x_err_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(position_x_errors[current_index], position_x_err_ders[current_index], position_x_err_ints[current_index]))
    update_2_vals_plot(position_y_art, position_y_errors[start_index:end_index], position_y_err_ders[start_index:end_index], "v {:.2f} d {:.2f}\ni {:.2f}".format(position_y_errors[current_index], position_y_err_ders[current_index], position_y_err_ints[current_index]))

    if not pause:
        frame_counter += 1
    return [text] + pitch_art + roll_art + height_art + yaw_art + pitch_roll_err_art + position_err_art + position_x_art + position_y_art

ani = FuncAnimation(fig, loop, interval=16, blit=True)

plt.show()
