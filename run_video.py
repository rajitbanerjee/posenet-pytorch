import torch
import cv2
import time
import argparse
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--input', type=str,
                    default='../../assets/videos/jumping_jacks.mp4')
parser.add_argument('--output', type=str,
                    default='../../src/posenet-py/output/')
parser.add_argument('--scale_factor', type=float, default=1.0)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    # def decode_fourcc(cc):
    #     return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    # fourcc = cv2.VideoWriter_fourcc(*decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)))

    # Read input video
    cap = cv2.VideoCapture(args.input)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_filepath = args.output + \
        args.input.split('/')[-1].split('.')[0] + '_out.mp4'
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        # End of video
        if type(input_image) == bool and not input_image:
            break

        # Pose estimation
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(
                input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        # Write frame to output video
        out.write(overlay_image)

        if not args.notxt:
            print()
            print("Results for frame #%d: " % frame_count)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' %
                          (posenet.PART_NAMES[ki], s, c))

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
