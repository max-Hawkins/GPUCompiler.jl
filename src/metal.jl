# implementation of the GPUCompiler interfaces for generating Metal code

## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    macos::VersionNumber=v"12.1.0"
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
        #callconv!(f, #=LLVM.API.LLVMMETALFUNCCallConv=# LLVM.API.LLVMCallConv(102))
        # XXX: this makes InstCombine erase kernel->func calls.
        #      do we even need this? why?
    end
end

function process_entry!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMMETALKERNELCallConv #=LLVM.API.LLVMCallConv(103)=#)
        # TODO: Fix argument types here??
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

        # TESTING: Adding llvm.module.flags
        # wchar_size = 4
        # wchar_key = "wchar_size"
        # wchar_md = Metadata[]
        # push!(wchar_md, Metadata(ConstantInt(Int32(1); ctx)))
        # push!(wchar_md, MDString("wchar_size"; ctx))
        # push!(wchar_md, Metadata(ConstantInt(Int32(4); ctx)))
        # wchar_md = MDNode(wchar_md; ctx)
        # LLVM.API.LLVMAddModuleFlag(mod, LLVM.API.LLVMModuleFlagBehavior(1), 
        #         Cstring(pointer(wchar_key)), Csize_t(length(wchar_key)),
        #         wchar_md)


        # function LLVMAddModuleFlag(M, Behavior, Key, KeyLen, Val)
        #     ccall((:LLVMAddModuleFlag, libllvm[]), Cvoid, (LLVMModuleRef, LLVMModuleFlagBehavior, Cstring, Csize_t, LLVMMetadataRef), M, Behavior, Key, KeyLen, Val)
        # end

        # Add air version metadata
        air_md = Metadata[]
        push!(air_md, Metadata(ConstantInt(Int32(2); ctx)))
        push!(air_md, Metadata(ConstantInt(Int32(4); ctx)))
        push!(air_md, Metadata(ConstantInt(Int32(0); ctx)))
        air_md = MDNode(air_md; ctx)
        push!(metadata(mod)["air.version"], air_md)

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
    # Do this for regular arguments in Metal.jl somehow??? Seems odd to fit it in here
    # Create metadata for argument buffer holding MtlDeviceArray
    # TODO: Check for duplicated argument buffer struct_type_info? Is this worth it?
    function process_arg(arg_infos, ty, i)
        arg_names = ["A", "B", "C"]
        arg_info = Metadata[]
        @info "In process arg: " i ty
        #=
        struct MtlDeviceArray{T,N,A} <: AbstractArray{T,N}
            shape::Dims{N} => N x Int64
            ptr::DeviceBuffer{T,A} =>
        =#
        #=
        !19 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.struct_type_info", !20, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Array", !"air.arg_name", !"arrA"}
        !20 = !{i32 0, i32 4, i32 1, !"uint", !"size", !"air.indirect_argument", !21, i32 8, i32 8, i32 0, !"float", !"data", !"air.indirect_argument", !22}
        !21 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"size"}
        !22 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
        =#

        # Create argument buffer type info
        struct_type_info = Metadata[]
        #=
        !20 = !{i32 0, i32 4, i32 1, !"uint", !"size", !"air.indirect_argument", !21, i32 8, i32 8, i32 0, !"float", !"data", !"air.indirect_argument", !22}
        !21 = !{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"size"}
        !22 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
        =#
        push!(struct_type_info, Metadata(ConstantInt(Int32(i-1); ctx))) # Argument index?
        push!(struct_type_info, Metadata(ConstantInt(Int32(8); ctx))) # Offset? TODO: Properly calculate
        push!(struct_type_info, Metadata(ConstantInt(Int32(1); ctx))) # TODO: What is this?
        push!(struct_type_info, MDString("ulong"; ctx)) # First field type TODO: Check long? and >1 dimension LLVM.UInt64Type()?
        push!(struct_type_info, MDString("shape"; ctx)) # First field name
        push!(struct_type_info, MDString("air.indirect_argument"; ctx))

        shape_field_info = Metadata[]
        push!(shape_field_info, Metadata(ConstantInt(Int32(0); ctx))) # Field index
        push!(shape_field_info, MDString("air.indirect_constant"; ctx))
        push!(shape_field_info, MDString("air.location_index"; ctx))
        push!(shape_field_info, Metadata(ConstantInt(Int32(0); ctx))) # Field index again?
        push!(shape_field_info, Metadata(ConstantInt(Int32(1); ctx))) # Address space TODO: Check and get properly
        push!(shape_field_info, MDString("air.arg_type_name"; ctx))
        push!(shape_field_info, MDString("ulong"; ctx)) # TODO: Check long?
        push!(shape_field_info, MDString("air.arg_name"; ctx))
        push!(shape_field_info, MDString("shape"; ctx))
        shape_field_info = MDNode(shape_field_info; ctx)

        push!(struct_type_info, shape_field_info) # First field info
        push!(struct_type_info, Metadata(ConstantInt(Int32(8); ctx))) # TODO: What is this?
        push!(struct_type_info, Metadata(ConstantInt(Int32(8); ctx))) # TODO: What is this?
        push!(struct_type_info, Metadata(ConstantInt(Int32(0); ctx))) # TODO: What is this?
        push!(struct_type_info, MDString("float"; ctx)) # Second field data type TODO: Properly get this
        push!(struct_type_info, MDString("ptr"; ctx)) # Second field name
        push!(struct_type_info, MDString("air.indirect_argument"; ctx))

        ptr_field_info = Metadata[]
        push!(ptr_field_info, Metadata(ConstantInt(Int32(1); ctx))) # Field index
        push!(ptr_field_info, MDString("air.buffer"; ctx))
        push!(ptr_field_info, MDString("air.location_index"; ctx))
        push!(ptr_field_info, Metadata(ConstantInt(Int32(1); ctx))) # Field index again?
        push!(ptr_field_info, Metadata(ConstantInt(Int32(1); ctx))) # Address space TODO: Check and get properly
        push!(ptr_field_info, MDString("air.read_write"; ctx)) # TODO: Check for const array
        push!(ptr_field_info, MDString("air.arg_type_size"; ctx))
        push!(ptr_field_info, Metadata(ConstantInt(Int32(4); ctx))) # TODO: Get properly
        push!(ptr_field_info, MDString("air.arg_type_align_size"; ctx))
        push!(ptr_field_info, Metadata(ConstantInt(Int32(4); ctx))) # TODO: Get properly Base.datatype_alignment(T)?
        push!(ptr_field_info, MDString("air.arg_type_name"; ctx))
        push!(ptr_field_info, MDString("float"; ctx)) # TODO: Get properly
        push!(ptr_field_info, MDString("air.arg_name"; ctx))
        push!(ptr_field_info, MDString("ptr"; ctx))
        ptr_field_info = MDNode(ptr_field_info; ctx)

        push!(struct_type_info, ptr_field_info) # Second field info
        struct_type_info = MDNode(struct_type_info; ctx)

        # Create the argument buffer main metadata
        push!(arg_info, Metadata(ConstantInt(Int32(i-1); ctx))) # Argument index
        push!(arg_info, MDString("air.indirect_buffer"; ctx)) # Keyword for argument buffer TODO:Check
        push!(arg_info, MDString("air.buffer_size"; ctx))
        push!(arg_info, Metadata(ConstantInt(Int32(sizeof(ty)); ctx))) # Buffer size TODO: Is this ok?
        push!(arg_info, MDString("air.location_index"; ctx))
        push!(arg_info, Metadata(ConstantInt(Int32(i-1); ctx))) # Argument index
        push!(arg_info, Metadata(ConstantInt(Int32(1); ctx))) # Address space? FIXME: Create dictionary of Metal.AS here too?
        # TODO: Check for const array and put to air.read
        push!(arg_info, MDString("air.read_write"; ctx))
        # !"air.arg_type_name", !"Array", !"air.arg_name", !"arrA"}
        push!(arg_info, MDString("air.struct_type_info"; ctx))
        push!(arg_info, struct_type_info) # Argument buffer type info
        push!(arg_info, MDString("air.arg_type_size"; ctx))
        push!(arg_info, Metadata(ConstantInt(Int32(sizeof(ty)); ctx))) # Arg type size TODO: Always same as buffer size for arg buffers?
        push!(arg_info, MDString("air.arg_type_align_size"; ctx))
        push!(arg_info, Metadata(ConstantInt(Int32(8); ctx))) # TODO: How to properly calculate this?
        push!(arg_info, MDString("air.arg_type_name"; ctx))
        push!(arg_info, MDString("MtlDeviceArray"; ctx)) # TODO: Figure out what to put here. Does it matter?
        push!(arg_info, MDString("air.arg_name"; ctx))
        push!(arg_info, MDString(arg_names[i]; ctx)) # TODO: How to get this? Does the compiler job have it somewhere?
        # Ignore unused flag for now
    
        arg_info = MDNode(arg_info; ctx)
        push!(arg_infos, arg_info)
        return nothing
    end


    entry = functions(mod)[entry_fn]
    ## argument info
    @info "-------- In add input arguments " job.source.tt
    arg_infos = Metadata[]
    # Regular arguments first
    for (i, arg_type) in enumerate(job.source.tt.parameters)
        process_arg(arg_infos, arg_type, i)
    end
    # Intrinsics last
    for (i, intr_fn) in enumerate(used_intrinsics)
        arg_info = Metadata[]
        # Put intrinsics at end of argument list
        push!(arg_info, Metadata(ConstantInt(Int32(length(parameters(entry))-i); ctx)))
        # TODO: Remove .uint from the intrinsic names
        push!(arg_info, MDString("air." * kernel_intrinsics[intr_fn].air_intr[1:end-5]; ctx))
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
