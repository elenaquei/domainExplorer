using Polynomials, TaylorSeries, ForwardDiff, LinearAlgebra
using Random, Distributions, Interpolations
using Plots

# Global paramter below is only used for the example boundary functions
epsilon = 0.5

function ray(t,x₀,v)
    return x₀ + t*v
end

function bbox_zeros(f;order=64)
    # Calculate all zeros of a vector-valued black-box function f of a single variable, assumed to be a polynomial up to some order.
    # f must be implemented with differentiable code.
    # COMMENT: This could be done without the use of e.g. TaylorSeries.jl. We could just interpolate f to get the polynomial representation instead. Only doing this because TaylorSeries.jl will give exact results pretty quickly, and I am lazy.
    p = taylor_expand(f,order=order)
    r = Array{Vector{ComplexF64}}(undef,length(p))
    for n in axes(r,1)
        c = p[n].coeffs
        j = order+1
        while c[j]==0 & j>0
            j -= 1
        end
        if j==0
            c = zeros(Float64,1)
        end
        r[n] = roots(Polynomial(c[1:j]))
    end
    return r
end

function get_intersection_times(x₀,v;g=myboundary,poly_order=64)
    return bbox_zeros(t->g(ray(t,x₀,v)),order=poly_order)
end

function spherical_to_cartesian(rϕ)
    n = length(rϕ)
    if n == 1
        return rϕ
    elseif n == 2
        x = zeros(n)
        x[1] = rϕ[1] * cos(rϕ[2])
        x[2] = rϕ[1] * sin(rϕ[2])
        return x
    end
    r = rϕ[1]
    x = zeros(n)
    x[1] = r * cos(rϕ[2])
    for i in 2:(n-1)
        x[i] = r*prod([ sin(rϕ[j]) for j in 2:i ])*cos(rϕ[i+1])
    end
    x[n] = r*prod([ sin(rϕ[j]) for j in 2:n ])
    return x
end

function sample_hypersphere(dimension,samples)
    pi_angles = rand(dimension-2,samples)*pi
    pi2_angles = rand(1,samples)*2pi
    radius = ones(1,samples)
    spherical_coordinates = vcat(radius,pi_angles,pi2_angles)
    return map(j->spherical_to_cartesian(spherical_coordinates[:,j]),1:samples)
end

function sample_hypersphere(dimension)
    return sample_hypersphere(dimension,1)[1]
end

function collision(x₀,v;g=myboundary,tolerance=1E-16,poly_order=64)
    time_vector = get_intersection_times(x₀,v;g=g,poly_order=poly_order)
    for n in axes(time_vector,1)
        time_vector[n] = time_vector[n][isreal.(time_vector[n])]                # Exclude non-real intersections
        time_vector[n] = time_vector[n][real.(time_vector[n]) .> tolerance]     # Keep only positive-time with time higher than tolerance
    end
    real_time_vector = real.(time_vector)
    first_intersection_boundary_index = -Inf
    first_intersection_time = Inf
    for m in axes(real_time_vector,1)
        if length(real_time_vector[m])>0
            i = argmin(real_time_vector[m])
            if real_time_vector[m][i]<first_intersection_time && tolerance<real_time_vector[m][i]
                first_intersection_time = real_time_vector[m][i]
                first_intersection_boundary_index = m
            end
        end
    end
    terminus = x₀ + v*first_intersection_time
    return terminus, first_intersection_boundary_index, first_intersection_time
end

function get_new_direction(x₀,boundary_index;g=myboundary)
    differential = ForwardDiff.jacobian(g,x₀)    # This is the only other part of the pipeline that requires the boundary function to be implemented with differentiable code. If we are willing to sacrifice a bit on rigour, we could just sample the hypersphere to get S like now, and do a check that all( g(x₀+tS) .< 0 ) for t>0 small enough, and if true, return S. 
    gradient = differential[boundary_index,:]
    S = sample_hypersphere(length(x₀))
    if dot(S,gradient)<0
        S = -S
    end
    return -S
end

function bounce_safe_mode(x₀,n_collisions;g=myboundary,poly_order=64)
    v₀ = sample_hypersphere(length(x₀))
    x_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
    v_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
    b_iterates = Vector{Int}(undef,n_collisions+1)
    t_iterates = Vector{Float64}(undef,n_collisions+1)
    x_iterates[1] = x₀
    v_iterates[1] = v₀
    b_iterates[1] = 0
    t_iterates[1] = 0
    n = 1
    while n <= n_collisions
        xn,bn,cn = collision(x_iterates[n],v_iterates[n];g=g,poly_order=poly_order)
        if bn<0 # Previous direction led to no collision; if domain is closed, likely caused by numerical error or bad initial condition.
            if n>1
                n -= 1
                v_iterates[n+1] = get_new_direction(x_iterates[n+1],b_iterates[n+1];g=g)
            else
                v_iterates[1] = sample_hypersphere(length(x₀))
            end
        else
            x_iterates[n+1],b_iterates[n+1],t_iterates[n+1] = xn,bn,cn
            v_iterates[n+1] = get_new_direction(x_iterates[n+1],b_iterates[n+1];g=g)
            n += 1
        end
    end
    return x_iterates,v_iterates,b_iterates,t_iterates
end

function bounce(x₀,n_collisions;g=myboundary,poly_order=64,safe_mode=false)
    # Runs the billiard with random reflection.
    # x₀: initial condition
    # n_collisions: number of collisions to compute (integer)
    # g: vector-valued function with domain having dim(x₀) such that the domain is equivalent to all( g(x) .<= 0 ). This code assumes g is polynomial, and in this case, outputs are theoretically correct.
    # safe_mode: bool; if true, resets on bad bounces. Can be necessary for domains with an extremely small "gap". 
    # poly_order: if g's components are multivariate polynomial, use the maximum order for best performance. Otherwise, I recommend leaving this as is. 
    if any(g(x₀) .> 0)
        error("Initial coordinate is not in the target domain.")
    end
    if safe_mode
        return bounce_safe_mode(x₀,n_collisions;g=myboundary)
    end
    v₀ = sample_hypersphere(length(x₀))
    x_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
    v_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
    b_iterates = Vector{Int}(undef,n_collisions+1)
    t_iterates = Vector{Float64}(undef,n_collisions+1)
    x_iterates[1] = x₀
    v_iterates[1] = v₀
    b_iterates[1] = 0
    t_iterates[1] = 0
    for n=1:n_collisions
        x_iterates[n+1],b_iterates[n+1],t_iterates[n+1] = collision(x_iterates[n],v_iterates[n];g=g,poly_order=poly_order)
        v_iterates[n+1] = get_new_direction(x_iterates[n+1],b_iterates[n+1];g=g)
    end
    return x_iterates,v_iterates,b_iterates,t_iterates
end

function metropolis(x₀,σ2,n_steps;g=myboundary,batchsize=1000)
    # A metropolis implementation. 
    # x₀ = initial coordinate
    # σ2 = variance
    if any(g(x₀) .>= 0)
        error("Initial coordinate to metropolis is not in the interior of target domain.")
    end
    path = zeros(Float64,length(x₀),n_steps+1)
    path[:,1] = copy(x₀)
    d = Normal(0,σ2)
    batch = rand(d,length(x₀),batchsize)
    b_index = 1
    for n=2:n_steps+1
        if b_index>batchsize
            batch = rand(d,length(x₀),batchsize)
            b_index = 1
        end
        path[:,n] = path[:,n-1] + batch[:,b_index]
        while any(g(path[:,n]) .> 0)
            if b_index>batchsize
                batch = rand(d,length(x₀),batchsize)
                b_index = 1
            end
            path[:,n] = path[:,n-1] + batch[:,b_index]
            b_index += 1
        end
        b_index += 1 
    end
    return path
end

function myboundary(x)
    return [
        sum(x.^2) - (1+epsilon^2);      #inside big sphere
        -sum(x.^2) + 1                  #outside smaller sphere
    ]
end

function myboundary_nested_sphere_in_cube(x)
    return [
        x .- (1+epsilon);   #inside big cube
        -x .- (1+epsilon);  #inside big cube
        -sum(x.^2) + 1      #outside smaller sphere
    ]
end

# Example:

x,v,b,t = bounce([1.05;0], 200, poly_order = 2)
X = hcat(x...)
plot(X[1,:], X[2,:])



# # Stuff below here is for a physical billiard; not going to be needed, we are assuming our domain has a Lipschitz boundary.

# function matrix_dot(A)
#     return sum(A.*A,dims=2)
# end

# function get_intersection_times_small_billiard_poly(x₀,v,ϵ;g=myboundary,Dg=D_myboundary,tolerance=1E-12)
#     time_vector = bbox_zeros(t->g(ray(t,x₀,v)).^2 - ϵ^2*matrix_dot(Dg(ray(t,x₀,v))) )   # Relaxation zeros
#     for boundary_index in axes(time_vector,1)                                                             
#         good_index = zeros(Bool,length(time_vector[boundary_index]))
#         for n in axes(time_vector[boundary_index],1)
#             t = time_vector[boundary_index][n]
#             good_index[n] = ( abs( g(ray(t,x₀,v))[boundary_index] + ϵ*norm( Dg(ray(t,x₀,v))[boundary_index,:] , 2) ) < tolerance )
#         end
#         time_vector[boundary_index] = time_vector[boundary_index][good_index]   # Remove extra zeros obtained by squared relaxation
#     end
#     return time_vector
# end

# function physical_collision(x₀,v,ϵ;g=myboundary,Dg=D_myboundary,tolerance=1E-14,exact_Newton=true,Newton_iters=10,Newton_convergence_tolerance=1E-10)
#     time_vector = get_intersection_times_small_billiard_poly(x₀,v,ϵ;g=g,Dg=Dg)  # Get relaxed collisions assuming small billiard approximation
#     for n in axes(time_vector,1)
#         time_vector[n] = time_vector[n][isreal.(time_vector[n])]                # Exclude non-real intersections
#         time_vector[n] = time_vector[n][real.(time_vector[n]) .> tolerance]     # Keep only positive-time with time higher than tolerance
#     end
#     real_time_vector = real.(time_vector)
#     first_intersection_boundary_index = 0
#     first_intersection_time = Inf
#     # Find the most likely collision -- that is, the first one -- and correct it with Newton's method.
#     for boundary_index in axes(real_time_vector,1)
#         for n in axes(real_time_vector[boundary_index],1)
#             if real_time_vector[boundary_index][n]<first_intersection_time
#                 first_intersection_time = real_time_vector[boundary_index][n]
#                 first_intersection_boundary_index = boundary_index
#             end
#         end
#     end
#     Y = physical_collision_correction(x₀,v,first_intersection_time,ϵ,first_intersection_boundary_index;g=g,iters=Newton_iters,convergence_tolerance=Newton_convergence_tolerance,exact_Newton=exact_Newton)
#     φ = Y[2:end-1]
#     terminus = x₀ + v*first_intersection_time
#     boundary_terminus = x₀ + first_intersection_time*v + ϵ*φ
#     return terminus, boundary_terminus, first_intersection_time, first_intersection_boundary_index
# end

# function physical_intersection_map(x₀,v,t,ϵ,φ,λ,boundary_index;g=myboundary)
#     evaluation = g(x₀ + t*v + ϵ*φ)[boundary_index]
#     differential = ForwardDiff.jacobian(g,x₀ + t*v + ϵ*φ)
#     gradient = differential[boundary_index,:]
#     return vcat(
#         [evaluation],
#         gradient - λ*φ,
#         [dot(φ,φ)-1]
#     )
# end

# function Newton(x₀,f;iters=10,convergence_tolerance=1E-10,exact_Newton=true)
#     J = ForwardDiff.jacobian(f,x₀)
#     invJ = inv(J)   
#     x = x₀
#     n=1
#     while norm(f(x),2)>convergence_tolerance && n<iters
#         x = x - invJ*f(x)
#         if exact_Newton
#             J = ForwardDiff.jacobian(f,x)
#             invJ = inv(J) 
#         end
#         n+=1
#     end
#     return x
# end

# function physical_collision_correction(x₀,v,t,ϵ,boundary_index;g=myboundary,iters=10,convergence_tolerance=1E-10,exact_Newton=true)
#     base_differential =  ForwardDiff.jacobian(g,x₀ + t*v)
#     φ₀ = base_differential[boundary_index,:]
#     λ = norm(φ₀,2)  # Initial guess for λ 
#     φ = φ₀/λ        # Initial guess for φ
#     Y₀ = vcat([t],φ,[λ])
#     return Newton(
#         Y₀,
#         Y->physical_intersection_map(x₀,v,Y[1],ϵ,Y[2:end-1],Y[end],boundary_index;g=g);
#         iters=iters,
#         exact_Newton=exact_Newton,
#         convergence_tolerance=convergence_tolerance
#     )
# end

# function physical_bounce(x₀,ϵ,n_collisions;g=myboundary,Dg=D_myboundary,tolerance=1E-14,exact_Newton=true,Newton_iters=10,Newton_convergence_tolerance=1E-10)
#     v₀ = sample_hypersphere(length(x₀))
#     billiard_central_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
#     boundary_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
#     v_iterates = Array{Vector{Float64}}(undef,n_collisions+1)
#     b_iterates = Vector{Int}(undef,n_collisions+1)
#     t_iterates = Vector{Float64}(undef,n_collisions+1)
#     billiard_central_iterates[1] = Float64.(x₀)
#     boundary_iterates[1] = Float64.(x₀)
#     v_iterates[1] = copy(v₀)
#     b_iterates[1] = 0
#     t_iterates[1] = 0
#     for n=1:n_collisions
#         billiard_central_iterates[n+1],boundary_iterates[n+1],t_iterates[n+1],b_iterates[n+1] = physical_collision(billiard_central_iterates[n],v_iterates[n],ϵ;g=g,Dg=Dg,tolerance=tolerance,exact_Newton=exact_Newton,Newton_iters=Newton_iters,Newton_convergence_tolerance=Newton_convergence_tolerance)
#         while any( g(billiard_central_iterates[n+1]) .> 0)  # bad bounce; reset direction and try again
#             v_iterates[n] = get_new_direction(boundary_iterates[n],b_iterates[n];g=g)
#             billiard_central_iterates[n+1],boundary_iterates[n+1],t_iterates[n+1],b_iterates[n+1] = physical_collision(billiard_central_iterates[n],v_iterates[n],ϵ;g=g,Dg=Dg,tolerance=tolerance,exact_Newton=exact_Newton,Newton_iters=Newton_iters,Newton_convergence_tolerance=Newton_convergence_tolerance)
#         end
#         v_iterates[n+1] = get_new_direction(boundary_iterates[n+1],b_iterates[n+1];g=g)
#     end
#     return billiard_central_iterates,v_iterates,b_iterates,t_iterates,boundary_iterates
# end   

# # Below here is benchmarking junk.

# struct VisitBox
#     dim::Int
#     width::Int
#     visited::Array
# end

# function count_visited(V::VisitBox)
#     return sum(V.visited)
# end

# function visit(x::Vector{Int},V::VisitBox)
#     shifted_x = x .+ V.width .+ 1
#     V.visited[CartesianIndex(Tuple(shifted_x))]=true
# end

# VisitBox(grid_diameter,bounding_radius,dim) = VisitBox(dim,floor.(Int,bounding_radius/grid_diameter),initialize_box(grid_diameter,bounding_radius,dim))

# function initialize_box(grid_diameter,bounding_radius,dim)
#     # Create dim-dimenisonal boolean array with one-dimensional size equal to 2*bounding_radius/grid_diameter+1
#     ind = floor.(Int,bounding_radius/grid_diameter)
#     return zeros(Bool,Tuple(ones(Int,dim)*(2*ind+1)))
# end

# function box_coordinate(point,diameter)
#     # Return the box of given diameter containing point, identified with a triple of integers of appropriate dimension.
#     return floor.(Int,point/diameter)
# end

# function subsample(path,subsamples)
#     diff_mat = diff(path,dims=2)
#     path_ss = Matrix{eltype(path)}(undef,size(path,1),(size(path,2)-1)*(subsamples) + 1)
#     for n=1:size(path,2)-1
#         path_ss[:,1+(n-1)*(subsamples):n*(subsamples)] = (path[:,n] .+ diff_mat[:,n].*(LinRange(0,1,subsamples+1)[1:end-1])')[:,:]
#     end
#     path_ss[:,end] = path[:,end]
#     return path_ss
# end

# function benchmark_metropolis(x₀,σ,n_steps,bounding_radius,grid_diameter;g=myboundary,batchsize=1000)
#     time = @elapsed begin
#         path = metropolis(x₀,σ,n_steps;g=g,batchsize=batchsize)
#     end
#     boxes = Vector{Vector{Int}}(undef,size(path,2))
#     for n in axes(boxes,1)
#         boxes[n] = box_coordinate(path[:,n],grid_diameter)
#     end
#     visited_boxes = VisitBox(grid_diameter,bounding_radius,length(x₀))
#     unique_boxes = Vector{Int}(undef,size(path,2))
#     for n in axes(path,2)
#         visit(box_coordinate(path[:,n],grid_diameter),visited_boxes)
#         unique_boxes[n] = count_visited(visited_boxes)
#     end
#     global_time = LinRange(0,time,size(path,2))
#     return boxes,unique_boxes,global_time
# end

# function benchmark_billiard(x₀,n_bounces,bounding_radius,grid_diameter;g=myboundary,n_subsamples=4)
#     time = @elapsed begin
#         path,_,_,_ = bounce(x₀,n_bounces;g=g)
#     end
#     path_cat = hcat(path...)
#     path_cat = subsample(path_cat,n_subsamples)
#     boxes = Vector{Vector{Int}}(undef,size(path_cat,2))
#     for n in axes(boxes,1)
#         boxes[n] = box_coordinate(path_cat[:,n],grid_diameter)
#     end
#     visited_boxes = VisitBox(grid_diameter,bounding_radius,length(x₀))
#     unique_boxes = Vector{Int}(undef,size(path_cat,2))
#     for n in axes(path_cat,2)
#         visit(box_coordinate(path_cat[:,n],grid_diameter),visited_boxes)
#         unique_boxes[n] = count_visited(visited_boxes)
#     end
#     global_time = LinRange(0,time,size(path_cat,2))
#     return boxes,unique_boxes,global_time
# end

# function orbit_sampler(x,interior_points)
#     X = hcat(x...)
#     dimension_space = size(X,1)
#     orbit_points = size(X,2)
#     sample = zeros(orbit_points-1,dimension_space,interior_points)
#     sample_out = zeros(dimension_space,interior_points*(orbit_points-1))
#     for m=1:orbit_points-1
#         sample[m,:,:] = X[:,m] .+ (X[:,m+1]-X[:,m]).*reshape(LinRange(0,1,interior_points),1,interior_points)
#         sample_out[:,1+(m-1)*interior_points:m*interior_points] = sample[m,:,:]
#     end
#     return sample_out
# end

# function benchmarktool(dim,g,n_metropolis,n_billiard,subsamples_billiard,runs,bounding_radius;grid_diameter=0.01)
#     metro_run = []
#     metro_interp = []
#     billiard_run = []
#     billiard_interp = []
#     time_min = Inf
#     for n = 1:runs
#         x = -(1+epsilon) .+ 2*(1+epsilon)*rand(dim)
#         while any(g(x) .> 0)
#             x = -(1+epsilon) .+ 2*(1+epsilon)*rand(dim)
#         end
#         metro_run = benchmark_metropolis(x,1,n_metropolis,bounding_radius,grid_diameter;g=g)
#         billiard_run = benchmark_billiard(x,n_billiard,bounding_radius,grid_diameter;g=g,n_subsamples=subsamples_billiard)
#         time_min = min(time_min, metro_run[3][end], billiard_run[3][end])
#         metro_interp = push!(metro_interp,linear_interpolation(metro_run[3],metro_run[2]))
#         billiard_interp = push!(billiard_interp,linear_interpolation(billiard_run[3],billiard_run[2]))
#         println("Run "*string(n)*" complete.")
#     end
#     t = Vector(LinRange(0.,time_min,100))
#     y_metro = zeros(100)
#     y_billiard = zeros(100)
#     for n=1:runs
#         y_metro += metro_interp[n](t)
#         y_billiard += billiard_interp[n](t)
#     end
#     return t, y_metro/runs, y_billiard/runs
# end
