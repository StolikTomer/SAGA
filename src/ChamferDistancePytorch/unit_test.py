import torch, time
import chamfer3D.dist_chamfer_3D
import chamfer_python

cham3D = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

from torch.autograd import Variable
from fscore import fscore

def test_chamfer(distChamfer, dim):
    points1 = torch.rand(4, 100, dim).cuda()
    points2 = torch.rand(4, 200, dim, requires_grad=True).cuda()
    dist1, dist2, idx1, idx2= distChamfer(points1, points2)

    loss = torch.sum(dist1)
    loss.backward()

    mydist1, mydist2, myidx1, myidx2 = chamfer_python.distChamfer(points1, points2)
    d1 = (dist1 - mydist1) ** 2
    d2 = (dist2 - mydist2) ** 2
    assert (
        torch.mean(d1) + torch.mean(d2) < 0.00000001
    ), "chamfer cuda and chamfer normal are not giving the same results"

    xd1 = idx1 - myidx1
    xd2 = idx2 - myidx2
    assert (
            torch.norm(xd1.float()) + torch.norm(xd2.float()) == 0
    ), "chamfer cuda and chamfer normal are not giving the same results"
    print(f"fscore :", fscore(dist1, dist2))
    print("Unit test passed")


def timings(distChamfer, dim):
    p1 = torch.rand(32, 2000, dim).cuda()
    p2 = torch.rand(32, 1000, dim).cuda()
    print("Timings : Start CUDA version")
    start = time.time()
    num_it = 100
    for i in range(num_it):
        points1 = Variable(p1, requires_grad=True)
        points2 = Variable(p2)
        mydist1, mydist2, idx1, idx2 = distChamfer(points1, points2)
        loss = torch.sum(mydist1)
        loss.backward()
    print(f"Ellapsed time forward backward is {(time.time() - start)/num_it} seconds.")


    print("Timings : Start Pythonic version")
    start = time.time()
    for i in range(num_it):
        points1 = Variable(p1, requires_grad=True)
        points2 = Variable(p2)
        mydist1, mydist2, idx1, idx2 = chamfer_python.distChamfer(points1, points2)
        loss = torch.sum(mydist1)
        loss.backward()
    print(f"Ellapsed time  forward backward  is {(time.time() - start)/num_it} seconds.")



dims = 3
print(f"testing Chamfer {dims}D")
test_chamfer(cham3D, dims)
timings(cham3D, dims)
