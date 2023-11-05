"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

import scipy.interpolate
from numpy.linalg import svd
from scipy.interpolate import griddata

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])

from matplotlib import pyplot as plt, image as mpimg


class Solution:
    """Implement Projective Homography and Panorama Solution."""

    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        num_match_pts = match_p_src.shape[1]
        A = np.array([], dtype=match_p_src.dtype)
        for i in range(num_match_pts):
            u = match_p_src[0, i]
            v = match_p_src[1, i]
            u_tag = match_p_dst[0, i]
            v_tag = match_p_dst[1, i]
            # constructing the system of equations
            u_equation = np.array([u, v, 1, 0, 0, 0, -u_tag * u, -u_tag * v, -u_tag])
            v_equation = np.array([0, 0, 0, u, v, 1, -v_tag * u, -v_tag * v, -v_tag])
            A = np.append(A, u_equation, axis=0)
            A = np.append(A, v_equation, axis=0)

        A = np.reshape(A, (2 * num_match_pts, 9))
        [u, s, v] = np.linalg.svd(A, full_matrices=True, compute_uv=True)
        homography_transform_vec = v[-1, :]
        homography_transform_mat = np.reshape(homography_transform_vec, (3, 3))
        return homography_transform_mat

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        forward_transformed_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        for u in range(src_image.shape[1]):
            for v in range(src_image.shape[0]):
                # defining the src pixel homogenous vector: u is x axis, v is y axis, 1 is the third element
                src_pixel = np.array([u, v, 1])
                # applying the projective transform on the src pixel coordinates
                transformed_pixel = np.matmul(homography, src_pixel)
                # converting the result homogenous coordinates to cartesian coordinates
                transformed_pixel = np.divide(transformed_pixel, transformed_pixel[2])
                # taking only the first two elements of the transformed coordinates - (u_tag, v_tag)
                transformed_cartesian_pixel = transformed_pixel[0:2]
                # rounding the coordinates so it matches a specific integer pixel in the destination image
                transformed_cartesian_pixel = np.round(transformed_cartesian_pixel).astype(int)
                # making sure we obtained pixels which are inside the destination image boundaries
                if (0 <= transformed_cartesian_pixel[0] < dst_image_shape[1]
                        and 0 <= transformed_cartesian_pixel[1] < dst_image_shape[0]):
                    # if the transformed coordinates "fall" outside the destination image boundaries - we ignore it.
                    forward_transformed_image[transformed_cartesian_pixel[1], transformed_cartesian_pixel[0], :] = src_image[v, u, :]
        return forward_transformed_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """

        # Creating a meshgrid of columns and rows
        u = np.arange(1, src_image.shape[1] + 1)
        v = np.arange(1, src_image.shape[0] + 1)
        uv, vv = np.meshgrid(u, v, indexing='xy')
        # Generating a matrix of size 3x(H*W) which stores the pixel locations in homogeneous coordinates
        homogeneous_matrix = np.zeros((3, src_image.shape[0] * src_image.shape[1]), dtype=int)
        homogeneous_matrix[0, :] = uv.flatten()
        homogeneous_matrix[1, :] = vv.flatten()
        homogeneous_matrix[2, :] = 1
        # Transforming the source homogeneous coordinates to the target homogeneous coordinates
        transformed_coordinates = np.matmul(homography, homogeneous_matrix)
        # Normalizing to obtain cartesian coordinates
        transformed_coordinates = np.divide(transformed_coordinates, transformed_coordinates[2, :])
        # Converting the coordinates into integer values and clip them according to the destination image size
        transformed_coordinates = np.round(transformed_coordinates).astype(int)
        valid_coordinates = np.bitwise_and(np.bitwise_and(0 <= transformed_coordinates[0, :], transformed_coordinates[0, :] < dst_image_shape[1]),
                                           np.bitwise_and(0 <= transformed_coordinates[1, :], transformed_coordinates[1, :] < dst_image_shape[0]))
        # Extracting the indices for clarity
        u_tag = transformed_coordinates[0, valid_coordinates]
        v_tag = transformed_coordinates[1, valid_coordinates]
        u_src = (uv-1).flatten()[valid_coordinates]
        v_src = (vv-1).flatten()[valid_coordinates]
        # Planting the pixels from the source image to the target image
        transformed_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        transformed_image[v_tag, u_tag, :] = src_image[v_src, u_src, :]
        return transformed_image


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        num_of_matching_points = match_p_src.shape[1]
        num_of_inliers = 0
        tot_error = 0
        for i in range(num_of_matching_points):
            src_pixel = np.array([match_p_src[0, i], match_p_src[1, i], 1])
            dst_pixel = np.array([match_p_dst[0, i], match_p_dst[1, i]])
            transformed_pixel = np.matmul(homography, src_pixel)
            # Converting the result homogenous coordinates to cartesian coordinates
            transformed_pixel = np.divide(transformed_pixel, transformed_pixel[2])
            # Taking only the first two elements of the transformed coordinates - (u_tag, v_tag)
            transformed_cartesian_pixel = transformed_pixel[:-1]
            # Calculating Euclidean error:
            error = np.sqrt(np.power(transformed_cartesian_pixel[0] - dst_pixel[0], 2) +
                            np.power(transformed_cartesian_pixel[1] - dst_pixel[1], 2))
            if int(error) <= max_err:
                num_of_inliers += 1
                tot_error += np.power(error, 2)
        if num_of_inliers == 0:
            dist_mse = 10 ** 9
        else:
            dist_mse = tot_error / num_of_inliers
        fit_percent = num_of_inliers / num_of_matching_points
        return fit_percent, dist_mse


    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        mp_src_meets_model = []
        mp_dst_meets_model = []
        num_of_matching_points = match_p_src.shape[1]
        for i in range(num_of_matching_points):
            src_point = np.array([match_p_src[0, i], match_p_src[1, i], 1])
            dst_point = np.array([match_p_dst[0, i], match_p_dst[1, i]])
            transformed_point = np.matmul(homography, src_point)
            # converting the result homogenous coordinates to cartesian coordinates
            transformed_point = np.divide(transformed_point, transformed_point[2])
            # taking only the first two elements of the transformed coordinates - (u_tag, v_tag)
            transformed_cartesian_point = transformed_point[:-1]
            # Calculating Euclidean error:
            error = np.sqrt(np.power(dst_point[0] - transformed_cartesian_point[0], 2)
                               + np.power(dst_point[1] - transformed_cartesian_point[1], 2))
            if int(error) <= max_err:
                mp_src_meets_model.append(src_point[:-1])
                mp_dst_meets_model.append(dst_point)

        return (np.array(mp_src_meets_model).transpose(), np.array(mp_dst_meets_model).transpose())


    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        p = 0.99  # parameter determining the probability of the algorithm to succeed
        d = 0.5  # the minimal probability of points which meets with the model
        n = 4  # number of points sufficient to compute the model
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        # Initializing error for checking the transform we will obtain.
        # This is the worst error we can obtain for the worst model (with 0 inliers)
        error = 10 ** 9

        indices_array = np.arange(match_p_src.shape[1])
        for i in range(k):
            # Randomly selecting n points
            selected_indices = np.random.choice(indices_array, n, replace=False)
            selected_src_points = match_p_src[:, selected_indices]
            selected_dst_points = match_p_dst[:, selected_indices]
            # Computing model using the points that have been selected
            homography_transform = self.compute_homography_naive(selected_src_points, selected_dst_points)
            # Finding inliers (all points with error < t)
            mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(homography_transform, match_p_src, match_p_dst, t)
            prob_pts_meets_model = mp_src_meets_model.shape[1]/match_p_src.shape[1]
            if prob_pts_meets_model >= d:
                # Recomputing model using all inliers found in the previous stage
                homography_transform = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                fit_percent, dist_mse = self.test_homography(homography_transform, mp_src_meets_model, mp_dst_meets_model, max_err)
                if dist_mse < error:
                    model = homography_transform
                    error = dist_mse
        return model


    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        # Creating a mesh-grid of columns and rows of the destination image
        u_dst = np.arange(1, dst_image_shape[1] + 1)
        v_dst = np.arange(1, dst_image_shape[0] + 1)
        uv_dst, vv_dst = np.meshgrid(u_dst, v_dst, indexing='xy')
        # Creating a set of homogenous coordinates for the destination image using the mesh-grid generated above.
        homogeneous_matrix = np.zeros((3, dst_image_shape[0] * dst_image_shape[1]), dtype=int)
        homogeneous_matrix[0, :] = uv_dst.flatten()
        homogeneous_matrix[1, :] = vv_dst.flatten()
        homogeneous_matrix[2, :] = 1
        # Computing the corresponding coordinates in the source image using the backward projective homography.
        transformed_pixels = np.matmul(backward_projective_homography, homogeneous_matrix)
        transformed_pixels = np.divide(transformed_pixels,  transformed_pixels[2, :])
        # Creating the mesh-grid of source image coordinates.
        u_src = np.arange(1, src_image.shape[1] + 1)
        v_src = np.arange(1, src_image.shape[0] + 1)
        uv_src, vv_src = np.meshgrid(u_src, v_src, indexing='xy')
        src_pixels = np.zeros((2, src_image.shape[0] * src_image.shape[1]), dtype=int)
        src_pixels[0, :] = (uv_src-1).flatten()
        src_pixels[1, :] = (vv_src-1).flatten()
        pixels_src_values = src_image[src_pixels[1, :], src_pixels[0, :], :]
        # Computing the bi-cubic interpolation of the projected coordinates.
        transformed_image = griddata(src_pixels.transpose(), pixels_src_values,
                                     transformed_pixels[:-1, :].transpose(), method='cubic', fill_value=0).astype('uint8')
        backward_warp = transformed_image.reshape(dst_image_shape)
        return backward_warp


    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num) + np.round(pad_right).astype(int) + np.round(pad_left).astype(int)
        panorama_rows_num = int(dst_rows_num) + np.round(pad_up).astype(int) + np.round(pad_down).astype(int)
        pad_struct = PadStruct(pad_up=np.round(pad_up).astype(int),
                               pad_down=np.round(pad_down).astype(int),
                               pad_left=np.round(pad_left).astype(int),
                               pad_right=np.round(pad_right).astype(int))
        return panorama_rows_num, panorama_cols_num, pad_struct


    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # Building the translation matrix from the pads
        translation_matrix = np.array([[1, 0, -1*pad_left], [0, 1, -1*pad_up], [0, 0, 1]])
        # Composing the backward homography and the translation matrix together
        homography_with_translation = np.matmul(backward_homography, translation_matrix)
        # Normalizing the transformation obtained
        transformation_elements = homography_with_translation.reshape((1, 9))
        transformation_elements = np.divide(transformation_elements, np.linalg.norm(transformation_elements))
        final_homography = transformation_elements.reshape((3, 3))
        return final_homography


    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # Computing the forward homography and the panorama shape.
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, forward_homography)
        panorama_shape = (panorama_rows_num, panorama_cols_num, 3)
        # Computing the backward homography.
        backward_homography = np.linalg.inv(forward_homography)
        # Adding the appropriate translation to the homography so that the source image will plant in place
        backward_homography_with_translation = self.add_translation_to_backward_homography(backward_homography,
                                                                                           pad_struct.pad_left, pad_struct.pad_up)
        # Computing the backward warping with the appropriate translation.
        backward_image = self.compute_backward_mapping(backward_homography_with_translation, src_image, panorama_shape)
        # Creating an empty panorama image and plant there the destination image
        panorama_image = np.pad(dst_image, ((pad_struct.pad_up, pad_struct.pad_down),
                                            (pad_struct.pad_left, pad_struct.pad_right), (0, 0)), 'constant', constant_values=0)
        # Placing the backward warped image in the indices where the panorama image is zero
        panorama_image = np.where(panorama_image == 0, backward_image, panorama_image)
        return np.clip(panorama_image, 0, 255).astype(np.uint8)

