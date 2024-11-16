"use client"
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '../components/Navbar';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"

const AdminPage = () => {
    const [trainers, setTrainers] = useState([]);
    const [appointments, setAppointments] = useState([]);
    const [users, setUsers] = useState([]);
    const router = useRouter();
    const [newTrainerData, setNewTrainerData] = useState({
        name: '', age: '', expYears: '', phone: '', price: '', imgUrl: '', type: 'Personal', description: ''
    });
    const [editTrainerData, setEditTrainerData] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    useEffect(() => {
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        if (!currentUser || !currentUser.isAdmin) {
            router.push('/login');
        }
        fetchTrainers();
        fetchAppointments();
        fetchUsers();
    }, [router]);

    const fetchTrainers = async () => {
        try {
            const response = await fetch('/api/trainers');
            const data = await response.json();
            setTrainers(data);
        } catch (error) {
            console.error('Error fetching trainers:', error);
        }
    };

    const fetchAppointments = async () => {
        try {
            const APresponse = await fetch('/api/appointments');
            const APdata = await APresponse.json();
            setAppointments(APdata);
        } catch (error) {
            console.error('Error fetching appointments:', error);
        }
    };

    const fetchUsers = async () => {
        try {
            const Uresponse = await fetch('/api/users');
            const userData = await Uresponse.json();
            setUsers(userData);
        } catch (error) {
            console.error('Error fetching users:', error);
        }
    };

    const handleNewTrainerChange = (e) => {
        setNewTrainerData({ ...newTrainerData, [e.target.name]: e.target.value });
    };

    const handleCreateTrainerSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('/api/trainers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...newTrainerData,
                    imgUrl: newTrainerData.imgUrl.split(',').map((url) => url.trim()),
                }),
            });
            if (response.ok) {
                fetchTrainers();
                setNewTrainerData({
                    name: '',email: '', password: '', age: '', expYears: '', phone: '', price: '', imgUrl: '', type: 'Personal', description: ''
                });
            } else {
                const errorData = await response.json();
                console.error('Error creating trainer:', errorData.message);
            }
        } catch (error) {
            console.error('Error creating trainer:', error);
        }
    };

    const handleDeleteTrainer = async (trainerId) => {
        if (window.confirm("Are you sure you want to delete this trainer?")) {
            try {
                const response = await fetch(`/api/trainers/${trainerId}`, { method: 'DELETE' });
                if (response.ok) {
                    fetchTrainers();
                } else {
                    alert("Failed to delete the trainer.");
                }
            } catch (error) {
                console.error("Error deleting trainer:", error);
                alert("An error occurred while deleting the trainer.");
            }
        }
    };

    const handleUpdateSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch(`/api/trainers/${editTrainerData._id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(editTrainerData),
            });
            if (response.ok) {
                fetchTrainers();
                setIsModalOpen(false);
            } else {
                const errorData = await response.json();
                console.error('Error updating trainer:', errorData.message);
            }
        } catch (error) {
            console.error('Error updating trainer:', error);
        }
    };

    const deleteAppointment = async (id) => {
        try {
            const response = await fetch(`/api/appointments/${id}`, { method: 'DELETE' });
            if (response.ok) {
                fetchAppointments();
            } else {
                const errorData = await response.json();
                console.error('Error deleting appointment:', errorData.message);
            }
        } catch (error) {
            console.error('Error deleting appointment:', error);
        }
    };

    return (
        <>
            <Navbar pageTitle={'Admin Panel'} />
            <div className="container mx-auto p-4">
                <h1 className="text-3xl font-bold mb-6">Admin Panel</h1>
                
                <Tabs defaultValue="trainers">
                    <TabsList className="mb-4">
                        <TabsTrigger value="trainers">Manage Trainers</TabsTrigger>
                        <TabsTrigger value="appointments">Manage Appointments</TabsTrigger>
                    </TabsList>

                    <TabsContent value="trainers">
                        <Card>
                            <CardHeader>
                                <CardTitle>Create New Trainer</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <form onSubmit={handleCreateTrainerSubmit} className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <Label htmlFor="name">Name</Label>
                                            <Input id="name" name="name" value={newTrainerData.name} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="email">Email</Label>
                                            <Input id="email" name="email" value={newTrainerData.email} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="password">Password</Label>
                                            <Input id="password" name="password" value={newTrainerData.password} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="age">Age</Label>
                                            <Input type="number" id="age" name="age" value={newTrainerData.age} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="expYears">Years of Experience</Label>
                                            <Input type="number" id="expYears" name="expYears" value={newTrainerData.expYears} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="phone">Phone Number</Label>
                                            <Input type="tel" id="phone" name="phone" value={newTrainerData.phone} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="price">Price</Label>
                                            <Input type="number" id="price" name="price" value={newTrainerData.price} onChange={handleNewTrainerChange} required />
                                        </div>
                                        <div className="space-y-2">
                                            <Label htmlFor="type">Type</Label>
                                            <Select name="type" value={newTrainerData.type} onValueChange={(value) => setNewTrainerData({ ...newTrainerData, type: value })}>
                                                <SelectTrigger>
                                                    <SelectValue placeholder="Select trainer type" />
                                                </SelectTrigger>
                                                <SelectContent>
                                                    <SelectItem value="Personal">Personal</SelectItem>
                                                    <SelectItem value="Group">Group</SelectItem>
                                                    <SelectItem value="CrossFit">CrossFit</SelectItem>
                                                    <SelectItem value="Physical therapy">Physical therapy</SelectItem>
                                                    <SelectItem value="Bodybuilding">Bodybuilding</SelectItem>
                                                    <SelectItem value="Athleticism">Athleticism</SelectItem>
                                                </SelectContent>
                                            </Select>
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="imgUrl">Image URLs (comma-separated)</Label>
                                        <Input id="imgUrl" name="imgUrl" value={newTrainerData.imgUrl} onChange={handleNewTrainerChange} />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="description">Description</Label>
                                        <Textarea id="description" name="description" value={newTrainerData.description} onChange={handleNewTrainerChange} required />
                                    </div>
                                    <Button type="submit">Create Trainer</Button>
                                </form>
                            </CardContent>
                        </Card>

                        <Card className="mt-6">
                            <CardHeader>
                                <CardTitle>Trainers List</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Trainer Name</TableHead>
                                            <TableHead>Type</TableHead>
                                            <TableHead>Actions</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {trainers.length > 0? (trainers.map(trainer => (
                                            <TableRow key={trainer._id}>
                                                <TableCell>{trainer.name}</TableCell>
                                                <TableCell>{trainer.type}</TableCell>
                                                <TableCell>
                                                    <Button variant="outline" className="mr-2" onClick={() => {setEditTrainerData(trainer); setIsModalOpen(true);}}>Update</Button>
                                                    <Button variant="destructive" onClick={() => handleDeleteTrainer(trainer._id)}>Delete</Button>
                                                </TableCell>
                                            </TableRow>))): (<TableRow><TableCell colSpan="3">No trainers found</TableCell></TableRow>)}
                                    </TableBody>
                                </Table>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="appointments">
                        <Card>
                            <CardHeader>
                                <CardTitle>Appointments List</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Trainer</TableHead>
                                            <TableHead>Date</TableHead>
                                            <TableHead>Time</TableHead>
                                            <TableHead>User</TableHead>
                                            <TableHead>Action</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {appointments.length > 0? (appointments.map((appointment) => (
                                            <TableRow key={appointment._id}>
                                                <TableCell>{appointment.trainer ? appointment.trainer.name : 'Unavailable'}</TableCell>
                                                <TableCell>{new Date(appointment.date).toLocaleDateString()}</TableCell>
                                                <TableCell>{new Date(appointment.date).toLocaleTimeString()}</TableCell>
                                                <TableCell>{appointment.user.username}</TableCell>
                                                <TableCell>
                                                    <Button variant="destructive" onClick={() => deleteAppointment(appointment._id)}>Delete</Button>
                                                </TableCell>
                                            </TableRow>))): (<TableRow><TableCell colSpan="3">No appointments found</TableCell></TableRow>)}
                                    </TableBody>
                                </Table>
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>

                <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Update Trainer: {editTrainerData?.name}</DialogTitle>
                        </DialogHeader>
                        <form onSubmit={handleUpdateSubmit} className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="edit-name">Name</Label>
                                    <Input id="edit-name" name="name" value={editTrainerData?.name} onChange={(e) => setEditTrainerData({ ...editTrainerData, name: e.target.value })} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="edit-age">Age</Label>
                                    <Input type="number" id="edit-age" name="age" value={editTrainerData?.age} onChange={(e) => setEditTrainerData({ ...editTrainerData, age: e.target.value })} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="edit-expYears">Years of Experience</Label>
                                    <Input type="number" id="edit-expYears" name="expYears" value={editTrainerData?.expYears} onChange={(e) => setEditTrainerData({ ...editTrainerData, expYears: e.target.value })} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="edit-phone">Phone Number</Label>
                                    <Input type="tel" id="edit-phone" name="phone" value={editTrainerData?.phone} onChange={(e) => setEditTrainerData({ ...editTrainerData, phone: e.target.value })} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="edit-price">Price</Label>
                                    <Input type="number" id="edit-price" name="price" value={editTrainerData?.price} onChange={(e) => setEditTrainerData({ ...editTrainerData, price: e.target.value })} required />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="edit-type">Type</Label>
                                    <Select name="type" value={editTrainerData?.type} onValueChange={(value) => setEditTrainerData({ ...editTrainerData, type: value })}>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select trainer type" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="Personal">Personal</SelectItem>
                                            <SelectItem value="Group">Group</SelectItem>
                                            <SelectItem value="CrossFit">CrossFit</SelectItem>
                                            <SelectItem value="Physical therapy">Physical therapy</SelectItem>
                                            <SelectItem value="Bodybuilding">Bodybuilding</SelectItem>
                                            <SelectItem value="Athleticism">Athleticism</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="edit-imgUrl">Image URLs (comma-separated)</Label>
                                <Input 
                                    id="edit-imgUrl" 
                                    name="imgUrl" 
                                    value={editTrainerData?.imgUrl.join(', ')} 
                                    onChange={(e) => setEditTrainerData({ ...editTrainerData, imgUrl: e.target.value.split(',').map(url => url.trim()) })} 
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="edit-description">Description</Label>
                                <Textarea 
                                    id="edit-description" 
                                    name="description" 
                                    value={editTrainerData?.description} 
                                    onChange={(e) => setEditTrainerData({ ...editTrainerData, description: e.target.value })} 
                                    required 
                                />
                            </div>
                            <DialogFooter>
                                <Button type="submit">Update Trainer</Button>
                            </DialogFooter>
                        </form>
                    </DialogContent>
                </Dialog>
            </div>
        </>
    );
};

export default AdminPage;