import os
import warnings

import colorcet as cc
import matplotlib.image as pyplot_img
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth

ALPHA = 0.5
IMAGES_FOLDER = 'images'
LINEWIDTH = 1.5
DOT_SIZE = 25

def plot_regions_and_decision_boundary(gradients, axis_min, axis_max, axis_steps, db_pieces, points=None,
    print_name=''):
    def shift_point(x):
        x -= axis_min
        x *= axis_steps / (axis_max - axis_min)
        return x

    exp_path = f'{IMAGES_FOLDER}/'
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    my_dpi = 1
    figsize = (axis_steps, axis_steps)
    linewidth = LINEWIDTH * axis_steps
    dot_size = DOT_SIZE * axis_steps**2

    if points is not None:
        origin = (points[0] + points[1] + points[2]) / 3.
        basis = orth(np.asarray([points[1] - points[0], points[2] - points[0]]).T)
        plot_points = []
        for point in points:
            x = np.linalg.lstsq(basis, point - origin, rcond=None)[0]
            plot_points.append(x)

        shifted_points = []
        for point in plot_points:
            shifted_points.append([shift_point(point[0]), shift_point(point[1])])

    # Plot linear regions
    plt.figure(figsize=figsize, dpi=my_dpi)
    plt.xlim(xmin=0, xmax=axis_steps)
    plt.ylim(ymin=0, ymax=axis_steps)
    img = plt.imshow(np.asarray(gradients).reshape([axis_steps, axis_steps]),
               origin='lower', interpolation='None', cmap=cc.cm.glasbey_hv)

    if points is not None:
        plt.scatter(np.asarray(shifted_points)[:, 0], np.asarray(shifted_points)[:, 1], marker='o', color='black',
            s=dot_size)

    plt.axis('off')
    plt.tight_layout()
    regions_filename = f'{exp_path}' + f'{print_name} ' + 'regions.png'
    plt.savefig(regions_filename)
    plt.close()

    ####################################################################################################################

    # Plot the decision boundary
    plt.figure(figsize=figsize, dpi=my_dpi)
    plt.xlim(xmin=0, xmax=axis_steps)
    plt.ylim(ymin=0, ymax=axis_steps)
    image = pyplot_img.imread(regions_filename)
    image = np.flipud(image)

    for x_id in range(axis_steps):
        for y_id in range(axis_steps):
            image[x_id][y_id][3] = ALPHA
    plt.imshow(image, origin='lower')

    if points is not None:
        plt.scatter(np.asarray(shifted_points)[:, 0], np.asarray(shifted_points)[:, 1], marker='o', color='black',
            s=dot_size)

    for piece in db_pieces:
        leq = piece.lhs_equalities[0]
        req = piece.rhs_equalities[0]
        xs = []
        for out_id, (lineq, rineq) in enumerate(zip(piece.lhs_inequalities, piece.rhs_inequalities)):
            try:
                A = np.asarray([leq, lineq])
                b = np.asarray([req, rineq])
                x = np.linalg.solve(A, b)
                ineqs = [l @ x <= r for l,r in zip(
                    piece.lhs_inequalities[:out_id] + piece.lhs_inequalities[out_id + 1:],
                    piece.rhs_inequalities[:out_id] + piece.rhs_inequalities[out_id + 1:])]
                inside = all(ineqs)
                if inside:
                    xs.append(x)
            except Exception as e:
                continue
            if len(xs) == 2:
                xs = np.asarray(xs)
                x1 = np.asarray([shift_point(x) for x in xs[:, 0]])
                x2 = np.asarray([shift_point(x) for x in xs[:, 1]])
                plt.plot(x1, x2, color='black', linestyle='solid', linewidth=linewidth)
            # The exact method does not always work because of numerical issues, so back it up with the approximate one
            else:
                x1 = np.linspace(axis_min, axis_max, axis_steps)
                x2 = np.linspace(axis_min, axis_max, axis_steps)
                x1, x2 = np.meshgrid(x1, x2)

                x_arr = np.vstack([x1.ravel(), x2.ravel()])
                piece_mask = np.asarray([l @ x_arr <= r for l,r in zip(
                    piece.lhs_inequalities, piece.rhs_inequalities)]).all(axis=-2)

                f = x_arr.T @ leq - req
                new_f = []
                for i, m in enumerate(piece_mask):
                    if m:
                        new_f.append(f[i])
                    else:
                        new_f.append(np.nan)

                f = np.asarray(new_f).reshape(axis_steps, axis_steps)

                x1 = np.asarray([shift_point(x) for x in x1])
                x2 = np.asarray([shift_point(x) for x in x2])

                # Ignore the exceptions that the contour function throws sometimes
                # because there are no contour levels in the x array with the used axis step
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    plt.contour(x1, x2, f, [0], colors='k', linestyles='solid', linewidths=linewidth)

    plt.axis('off')
    plt.tight_layout()
    filename = f'{exp_path}' + f'{print_name} ' + 'decision boundary.png'
    plt.savefig(filename)
    plt.close()
