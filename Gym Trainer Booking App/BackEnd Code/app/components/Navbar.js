"use client";
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Menu } from "lucide-react";

function Navbar({ pageTitle }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const signOut = () => {
    localStorage.removeItem('currentUser');
    setUser(null);
    router.push('/login');
  };

  const isAdmin = user && user.isAdmin;
  const isTrainer = user && user.isTrainer;

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  return (
    <nav className="bg-gray-800 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">
          {pageTitle || 'Home'}
        </Link>
        <div className="hidden md:flex space-x-4 items-center">
          {user ? (
            <>
              <span className="text-sm">Hello {user.username}</span>
              {pageTitle !== 'My Appointments' && !isAdmin && !isTrainer && (
                <Button variant="ghost" asChild>
                  <Link href="/appointments">View Appointments</Link>
                </Button>
              )}
              {pageTitle === 'My Appointments' && (
                <Button variant="ghost" asChild>
                  <Link href="/home">Home</Link>
                </Button>
              )}
              <Button variant="ghost" onClick={signOut}>Sign Out</Button>
            </>
          ) : (
            <>
              <Button variant="ghost" asChild>
                <Link href="/register">Register</Link>
              </Button>
              <Button variant="ghost" asChild>
                <Link href="/login">Login</Link>
              </Button>
            </>
          )}
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild className="md:hidden">
            <Button variant="ghost" size="icon">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle menu</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>{user ? `Hello ${user.username}` : 'Menu'}</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {user ? (
              <>
                {pageTitle !== 'My Appointments' && !isAdmin && !isTrainer && (
                  <DropdownMenuItem asChild>
                    <Link href="/appointments">View Appointments</Link>
                  </DropdownMenuItem>
                )}
                {pageTitle === 'My Appointments' && (
                  <DropdownMenuItem asChild>
                    <Link href="/home">Home</Link>
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={signOut}>Sign Out</DropdownMenuItem>
              </>
            ) : (
              <>
                <DropdownMenuItem asChild>
                  <Link href="/register">Register</Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/login">Login</Link>
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </nav>
  );
}

export default Navbar;