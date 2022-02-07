# implementation of the GPUCompiler interfaces for generating Metal code

## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    macos::VersionNumber=v"12.0.0"
end

function Base.hash(target::MetalCompilerTarget, h::UInt)
    hash(target.macos, h)
end

source_code(target::MetalCompilerTarget) = "metal"

llvm_triple(target::MetalCompilerTarget) = "air64-apple-macosx$(target.macos)"

function llvm_machine(target::MetalCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple=triple)

    TargetMachine(t, triple)
end

llvm_datalayout(target::MetalCompilerTarget) =
    "e-p:64:64:64"*
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
    "-f32:32:32-f64:64:64"*
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"*
    "-n8:16:32"


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{MetalCompilerTarget}) = "metal-macos$(job.target.macos)"

function process_module!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module)
    # calling convention
    for f in functions(mod)
        #callconv!(f, #=LLVM.API.LLVMMetalFUNCCallConv=# LLVM.API.LLVMCallConv(102))
        # XXX: this makes InstCombine erase kernel->func calls.
        #      do we even need this? why?
    end
end

function process_entry!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # calling convention
        callconv!(entry, #=LLVM.API.LLVMMetalKERNELCallConv=# LLVM.API.LLVMCallConv(103))
    end

    return entry
end

# TODO: why is this done in finish_module? maybe just in process_entry?
function finish_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.source.kernel
        add_input_arguments!(job, mod, entry)
    end

    return functions(mod)[entry_fn]
end

const kernel_intrinsics = Dict()
for intr in [
        "dispatch_quadgroups_per_threadgroup", "dispatch_simdgroups_per_threadgroup",
        "quadgroup_index_in_threadgroup", "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup", "simdgroups_per_threadgroup",
        "thread_index_in_quadgroup", "thread_index_in_simdgroup", "thread_index_in_threadgroup",
        "thread_execution_width", "threads_per_simdgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("i16",   "ushort",  UInt16),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("v2i32", "uint2", NTuple{2, VecElement{UInt32}}),
        ("v3i32", "uint3", NTuple{3, VecElement{UInt32}}),
        ("i16",   "ushort",  UInt16),
        ("v2i16", "ushort2", NTuple{2, VecElement{UInt16}}),
        ("v3i16", "ushort3", NTuple{3, VecElement{UInt16}}),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end

function add_input_arguments!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                              entry::LLVM.Function)
    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    # figure out which intrinsics are used and need to be added as arguments
    used_intrinsics = filter(keys(kernel_intrinsics)) do intr_fn
        haskey(functions(mod), intr_fn)
    end |> collect
    isempty(used_intrinsics) && return false
    nargs = length(used_intrinsics)

    # add the arguments to every function
    worklist = filter(!isdeclaration, collect(functions(mod)))
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = LLVM.name(f)
        ft = eltype(llvmtype(f))
        LLVM.name!(f, fn * ".orig")

        # create a new function
        new_param_types = LLVMType[parameters(ft)...]
        for intr_fn in used_intrinsics
            llvm_typ = convert(LLVMType, kernel_intrinsics[intr_fn].julia_typ; ctx)
            push!(new_param_types, llvm_typ)
        end
        new_ft = LLVM.FunctionType(return_type(ft), new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        linkage!(new_f, linkage(f))
        for (arg, new_arg) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_arg, LLVM.name(arg))
        end
        for (intr_fn, new_arg) in zip(used_intrinsics, parameters(new_f)[end-nargs+1:end])
            LLVM.name!(new_arg, kernel_intrinsics[intr_fn].air_name)
        end

        workmap[f] = new_f
    end

    # clone and the function bodies.
    # we don't need to rewrite anything as the arguments are added last.
    for (f, new_f) in workmap
        # use a value mapper for rewriting function arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_param, LLVM.name(param))
            value_map[param] = new_param
        end

        # use a value materializer for replacing uses of the function in constants
        function materializer(val)
            if val isa LLVM.ConstantExpr && opcode(val) == LLVM.API.LLVMPtrToInt
                src = operands(val)[1]
                if haskey(workmap, src)
                    return LLVM.const_ptrtoint(workmap[src], llvmtype(val))
                end
            end
            return val
        end

        # NOTE: we need global changes because LLVM 12 wants to clone debug metadata
        clone_into!(new_f, f; value_map, materializer,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # we can't remove this function yet, as we might still need to rewrite any called,
        # but remove the IR already
        empty!(f)
    end

    # drop unused constants that may be referring to the old functions
    # XXX: can we do this differently?
    for f in worklist
        for use in uses(f)
            val = user(use)
            if val isa LLVM.ConstantExpr && isempty(uses(val))
                LLVM.unsafe_destroy!(val)
            end
        end
    end

    # update other uses of the old function, modifying call sites to pass the arguments
    function rewrite_uses!(f, new_f)
        # update uses
        Builder(ctx) do builder
            for use in uses(f)
                val = user(use)
                callee_f = LLVM.parent(LLVM.parent(val))
                if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                    # forward the arguments
                    position!(builder, val)
                    new_val = if val isa LLVM.CallInst
                        call!(builder, new_f, [arguments(val)..., parameters(callee_f)[end-nargs+1:end]...], operand_bundles(val))
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.ConstantExpr && opcode(val) == LLVM.API.LLVMBitCast
                    # XXX: why isn't this caught by the value materializer above?
                    target = operands(val)[1]
                    @assert target == f
                    new_val = LLVM.const_bitcast(new_f, llvmtype(val))
                    rewrite_uses!(val, new_val)
                    # we can't simply replace this constant expression, as it may be used
                    # as a call, taking arguments (so we need to rewrite it to pass the input arguments)

                    # drop the old constant if it is unused
                    # XXX: can we do this differently?
                    if isempty(uses(val))
                        LLVM.unsafe_destroy!(val)
                    end
                else
                    error("Cannot rewrite unknown use of function: $val")
                end
            end
        end
    end
    for (f, new_f) in workmap
        rewrite_uses!(f, new_f)
        @assert isempty(uses(f))
        unsafe_delete!(mod, f)
    end

    # replace uses of the intrinsics with references to the input arguments
    for (i, intr_fn) in enumerate(used_intrinsics)
        intr = functions(mod)[intr_fn]
        for use in uses(intr)
            val = user(use)
            callee_f = LLVM.parent(LLVM.parent(val))
            if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                replace_uses!(val, parameters(callee_f)[end-nargs+i])
            else
                error("Cannot rewrite unknown use of function: $val")
            end

            @assert isempty(uses(val))
            unsafe_delete!(LLVM.parent(val), val)
        end
        @assert isempty(uses(intr))
        unsafe_delete!(mod, intr)
    end

    # add metadata
    entry = functions(mod)[entry_fn]
    ## argument info
    arg_infos = Metadata[]
    for (i, intr_fn) in enumerate(used_intrinsics)
        arg_info = Metadata[]
        push!(arg_info, Metadata(ConstantInt(Int32(length(parameters(entry))-i); ctx)))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_intr; ctx))
        push!(arg_info, MDString("air.arg_type_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_typ; ctx))
        push!(arg_info, MDString("air.arg_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_name; ctx))
        arg_info = MDNode(arg_info; ctx)
        push!(arg_infos, arg_info)
    end
    arg_infos = MDNode(arg_infos; ctx)
    ## stage info
    stage_infos = Metadata[]
    stage_infos = MDNode(stage_infos; ctx)

    kernel_md = MDNode([entry, stage_infos, arg_infos]; ctx)
    push!(metadata(mod)["air.kernel"], kernel_md)

    return true
end
