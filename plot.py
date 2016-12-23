import matplotlib.pyplot
import numpy

matplotlib.pyplot.ion()


class prediction_error_plot:

    moving_avg_points = 16
    line_x_avg = []
    line_x = []
    line_y = []

    def __init__(self):

        self.line_x = []
        self.line_avg_x = []
        self.line_avg_y = []
        self.line_y = []

    def update(self, x, y):

        self.line_x.append(x)
        self.line_y.append(y)

        if len(self.line_x) >= self.moving_avg_points:
            self.line_avg_y.append(numpy.sum(self.line_y[-self.moving_avg_points:]) / self.moving_avg_points )
            self.line_avg_x.append(x)

        matplotlib.pyplot.plot(self.line_x, self.line_y, 'r-', linewidth = 1.0)
        matplotlib.pyplot.plot(self.line_avg_x, self.line_avg_y, 'b--', linewidth = 1.0)

        matplotlib.pyplot.axis(ymin = 0.0, ymax = 100.0)
        matplotlib.pyplot.ylabel('Learning error')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.pause(0.05)